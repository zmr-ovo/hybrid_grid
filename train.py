import os
import math
import argparse
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
from model import *
from util import *

def setup_logging(log_dir:str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清空旧 Handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 统一时间显示格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 文件日志
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'train.log'),mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    # 控制台日志
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    return logger

def setup_tb(log_dir:str):
    # TensorBoard
    tb_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)

def save_checkpoint(model, optimizer, epoch, best_psnr, path:str):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_psnr': best_psnr,
    }
    torch.save(state, path)

def train(args):
    # 初始化
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.out_dir, timestamp, f"{args.exp_name}")
    logger = setup_logging(log_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    args.warmup = int(args.warmup * args.epochs)

    # 数据集
    train_dataset = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=tuple(args.fixed_res),
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        frame_interval=args.frame_interval
    )

    val_dataset = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=tuple(args.fixed_res),
        frame_interval=args.frame_interval,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # 模型
    model = HybridGridNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = NervLoss(loss_type=args.loss_type, device=device)

    data_size = len(train_dataset)

    # 参数统计
    logging.info("\n" + "="*50 + " 模型结构 " + "="*50)
    logging.info(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    param_size = total_params / 1e6
    
    grid_encoder_params = sum(p.numel() for p in model.grid_encoder.parameters())
    level_params = []
    for i, level in enumerate(model.grid_encoder.levels):
        level_param = sum(p.numel() for p in level.parameters())
        level_params.append(level_param)
        logging.info(f"Grid Level {i}: {level_param:,} params | Resolution: {level.res_x}x{level.res_y}x{level.res_t}")
    
    pe_encoder_params = sum(p.numel() for p in model.pe_encoder.parameters())
    gate_params = sum(p.numel() for p in model.gate_grid.parameters()) + sum(p.numel() for p in model.gate_pe.parameters())
    mod_params = sum(p.numel() for p in model.time_mod.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())   
    
    logging.info("="*50 + " 详细参数统计 " + "="*50)
    logging.info(f"Grid: {grid_encoder_params:,} ({grid_encoder_params/total_params:.1%})")
    logging.info(f"PE: {pe_encoder_params:,} ({pe_encoder_params/total_params:.1%})")
    logging.info(f"Gate: {gate_params:,} ({gate_params/total_params:.1%})")
    logging.info(f"Mod: {mod_params:,} ({mod_params/total_params:.1%})")
    logging.info(f"Decoder: {decoder_params:,} ({decoder_params/total_params:.1%})")
    
    logging.info("="*50 + " 参数量汇总 " + "="*50)
    logging.info(f"总参数: {total_params:,}") 
    logging.info(f"参数大小: {param_size:.2f} MB")
    logging.info("="*107 + "\n")
    
    logging.info("\n" + "="*50 + " 实验参数 " + "="*50)
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("=" * 50 + "\n")

    # 评估
    if args.eval_only:
        logging.info("========== Eval-only ==========")
        eval_dir = os.path.join(log_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        val_psnr, val_msssim = evaluate(model, val_loader, device, args, epoch=-1, save_dir=eval_dir)

        logging.info(
            f"[Eval-only] 验证集结果 → PSNR: {val_psnr:.2f} dB | "
            f"MS-SSIM: {val_msssim:.4f} ")

        # 同步保存结果到 txt 文件
        result_path = os.path.join(eval_dir, "eval_results.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("========== Eval-only ==========\n")
            f.write(f"模型路径: {args.resume if getattr(args, 'resume', None) else '当前模型'}\n")
            f.write(f"PSNR: {val_psnr:.2f} dB\n")
            f.write(f"MSSSIM: {val_msssim:.4f}\n")
        logging.info(f"评估结果已保存到: {result_path}")

        return

        
    # 恢复训练（如果存在检查点）
    start_epoch = 0
    best_psnr = 0.0  
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            start_epoch = checkpoint.get("epoch", -1) + 1
            best_loss = checkpoint.get("loss", float("inf"))
            best_psnr = checkpoint.get("best_psnr", 0.0)

            logging.info(f"成功恢复检查点：{args.resume}(从epoch {start_epoch}继续训练)")
        else:
            logging.warning(f"未找到检查点：{args.resume}，重新开始训练")


    # 训练
    is_train_best = False
    train_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_msssim = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            
            coords = batch['coords'].to(device)
            pixels = batch['pixels'].to(device)
            
            preds = model(coords)
            image_loss = loss_fn(preds, pixels)
            total_loss = image_loss
            
            lr = adjust_lr(optimizer, epoch % args.epochs, batch_idx, data_size, args)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 计算指标
            psnr = psnr_fn(preds, pixels)
            msssim = msssim_fn(preds, pixels, device)
            
            epoch_loss += total_loss.item()
            epoch_psnr += psnr
            epoch_msssim += msssim
            
            # step日志
            if batch_idx % args.log_interval == 0 or batch_idx == data_size - 1:
                psnr_tmp = epoch_psnr / (batch_idx + 1)
                msssim_tmp = epoch_msssim / (batch_idx + 1)
                logging.info(
                    f"epoch [{epoch+1}/{args.epochs}] "
                    f"batch [{batch_idx+1}/{data_size}] "
                    f"loss: {total_loss.item():.4f} "
                    f"psnr: {psnr_tmp:.2f} dB  "
                    f"msssim: {msssim_tmp:.4f}"
                )
        
        # Epoch统计
        avg_loss = epoch_loss / data_size
        avg_psnr = epoch_psnr / data_size
        avg_msssim = epoch_msssim / data_size

        epoch_time = time.time() - epoch_start
        
        is_train_best = avg_psnr > best_psnr
        best_psnr = avg_psnr if avg_psnr > best_psnr else best_psnr

        # 记录到 TensorBoard
        tb_writer.add_scalar("train/loss", avg_loss, epoch + 1)
        tb_writer.add_scalar("train/psnr", avg_psnr, epoch + 1)
        tb_writer.add_scalar("train/msssim", avg_msssim, epoch + 1)
        tb_writer.add_scalar("time/epoch_sec", epoch_time, epoch + 1)
        
        logging.info(
            f"======  Epoch {epoch+1} 完成 ======"
            f"LOSS: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB | "
            f"BEST: {best_psnr:.2f} dB | "
            f"MSSSIM: {avg_msssim:.4f} | "
            f"TIME: {epoch_time:.2f}秒 | "
            f"LR: {lr:.3e}"
        )
        
        # 保存检查点
        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(log_dir, 'train_latest.pth')
            save_checkpoint(model, optimizer, epoch, best_psnr, checkpoint_path)
                    
        # 更新最佳模型
        if is_train_best:
            best_path = os.path.join(log_dir, 'train_best.pth')
            save_checkpoint(model, optimizer, epoch, best_psnr, best_path)  # 传入定义好的优化器
            
        # evaluate
        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_loader, device, epoch, save_dir=os.path.join(log_dir, 'visualize'))
            val_end_time = datetime.now()
            val_time = (val_end_time - val_start_time).total_seconds()

            logging.info(
                f"== Epoch {epoch+1} 测试完成 == "
                f"PSNR: {val_psnr:.2f} dB | "
                f"BEST: {best_psnr:.2f} dB | "
                f"MSSSIM: {val_msssim:.4f} | "
                f"TIME: {val_time:.2f}秒 | "
            )
    total_time = time.time() - train_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    logging.info(
        f"训练完成，总耗时: {hours}小时 {minutes}分钟 {seconds}秒 "
    )
    tb_writer.add_scalar("time/total_sec", total_time, 0)
    tb_writer.close()
        
@torch.no_grad()        
def evaluate(model, val_loader, device, epoch, save_dir='visualize'):
    total_psnr = 0.0
    total_msssim = 0.0
    time_list = []

    os.makedirs(save_dir, exist_ok=True) 

    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        coords = batch['coords'].to(device, non_blocking=True)  # (B,3,H,W)
        pixels = batch['pixels'].to(device, non_blocking=True)  # (B,3,H,W)
        
        torch.cuda.synchronize()  
        start_time = time.time()

        pred = model(coords)
        time_list.append(time.time() - start_time)
        
        if args.dump_images:
            B = pred.size(0)
            for i in range(B):
                sample_id = batch_idx * B + i

                pred_img = pred[i].detach().cpu()
                save_image(pred_img, os.path.join(save_dir, f'pred_{sample_id:05d}.png'))

                if epoch == args.epochs - 1:
                    gt_img = pixels[i].detach().cpu()
                    save_image(gt_img, os.path.join(save_dir, f'gt_{sample_id:05d}.png'))

        psnr = psnr_fn(pred, pixels)
        msssim = msssim_fn(pred, pixels, device)
        total_psnr += psnr
        total_msssim += msssim
    
        if batch_idx % args.log_interval == 0 or batch_idx == len(val_loader) - 1:

            fps = (batch_idx + 1) / sum(time_list)            
            psnr_tmp = total_psnr / (batch_idx + 1)
            msssim_tmp = total_msssim / (batch_idx + 1)

            logging.info(
                f"Step [{batch_idx + 1}/{len(val_loader)}] | " 
                f"Val PSNR: {psnr_tmp:.2f} dB | "
                f"Val MSSSIM: {msssim_tmp:.4f} | "
                f"FPS: {fps:.2f}"
            )
        
    model.train()
    avg_psnr = total_psnr / len(val_loader)
    avg_msssim = total_msssim / len(val_loader)
    return avg_psnr, avg_msssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练脚本")
    
    parser.add_argument('-d', '--data_root', type=str, required=True,
                       help='训练帧文件夹路径')
    
    # 训练参数
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='训练批次大小 (default: 8)')
    parser.add_argument('--epochs', type=int, default=1200,
                        help='总训练轮数 (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率 (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='优化器权重衰减 (default: 1e-6)')
    parser.add_argument('--lr_type', type=str, default='cosine', 
                        help='learning rate type, default=cosine')
    parser.add_argument('--warmup', type=float, default=0.2, 
                    help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--eval_only', action='store_true', 
                    help='运行评估模式（不再训练）')
    parser.add_argument('--fixed_res', type=int, nargs=2, default=[720, 1280],
                        help='评估分辨率 [H W] (default: 720 1280)')

    # 模型参数
    parser.add_argument('--grid_levels', type=int, default=12,
                      help='网格层级数 (default: 12)')
    parser.add_argument('--grid_feat_dim', type=int, default=4,
                      help='每级网格的特征维度 (default: 4)')
    parser.add_argument('--base_resolution', type=int, default=9,
                      help='基础分辨率 (default: 9)')
    parser.add_argument('--finest_resolution', type=int, default=32,
                      help='最精细分辨率 (default: 32)')
    parser.add_argument('--aspect_ratio', type=int, nargs=2, default=[16, 9],
                      help='宽高比 (default: 16 9)')
    parser.add_argument('--time_scale', type=float, default=0.5,
                      help='时间维度缩放因子 (default: 0.5)')
    parser.add_argument('--pe_freq', type=int, default=10,
                      help='位置编码频率数 (default: 10)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                      help='隐藏层维度 (default: 512)')
    
    # 数据参数
    parser.add_argument('--base_res', type=int, nargs=2, default=[720, 1280],
                      help='基准分辨率 [H W] (default: 720 1280)')
    parser.add_argument('--min_scale', type=float, default=0.5,
                      help='最小分辨率缩放因子 (default: 0.25)')
    parser.add_argument('--max_scale', type=float, default=1.2,
                      help='最大分辨率缩放因子 (default: 1.2)')
    parser.add_argument('--frame_interval', type=int, default=1,
                      help='帧采样间隔 (default: 1)')
    
    # 系统参数
    parser.add_argument('--out_dir', type=str, default='./experiments',
                      help='检查点保存路径 (default: ./experiments)')
    parser.add_argument('--exp_name', type=str, default='dynamic_nerv',
                      help='实验名称 (default: dynamic_nerv)')
    parser.add_argument('--save_interval', type=int, default=50,
                      help='保存间隔（epoch） (default: 50)')
    parser.add_argument('--log_interval', type=int, default=50,
                      help='日志间隔（batch） (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                      help='恢复训练的检查点路径')
    parser.add_argument('--eval_freq', type=int, default=50, 
                      help='评估频率（epoch）')
    parser.add_argument('--dump_images', action='store_true',
                      default=False, help='dump the prediction images')
    parser.add_argument('--loss_type', type=str, default='L2', 
                        choices=['L2', 'L1', 'SSIM', 'Fusion1', 'Fusion2', 
                                'Fusion3', 'Fusion4', 'Fusion5', 'Fusion6', 
                                'Fusion7', 'Fusion8', 'Fusion9', 'Fusion10', 
                                'Fusion11', 'Fusion12'],
                        help='损失函数类型 (default: L2)')
    

    # ...
    args = parser.parse_args()
    
    train(args)
