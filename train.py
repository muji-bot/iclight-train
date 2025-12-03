import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import PIL.Image
import numpy as np
import os
import json
from glob import glob
import random
import gc
from torchvision import transforms
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation


# 1. 环境编码器 - 必须先定义
class EnvironmentEncoder(nn.Module):
    def __init__(self, output_dim=2304):
        super().__init__()
        # 减小网络规模
        self.mlp = nn.Sequential(
            nn.Linear(32 * 32 * 3, 2048),  # 减小隐藏层大小
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_dim)
        )

    def forward(self, environment_map):
        batch_size = environment_map.shape[0]
        x = environment_map.reshape(batch_size, -1)
        x = self.mlp(x)
        x = x.reshape(batch_size, 3, 768)
        return x

# 2. 增强的数据集类
class ICLightDataset(Dataset):
    def __init__(self, data_dir="./training_images", image_size=512, max_samples=200):
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_samples = max_samples
        self.image_paths = self.find_images()[:max_samples]

        # 数据增强变换
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # 光照相关的提示词
        self.lighting_prompts = [
            "professional studio lighting", "natural sunlight illumination",
            "dramatic cinematic lighting", "soft ambient light",
            "warm golden hour lighting", "cool blue hour lighting",
            "harsh direct lighting", "soft diffused lighting",
            "backlit silhouette", "side lighting with shadows",
            "top down lighting", "rim lighting effect",
            "moody low key lighting", "bright high key lighting",
            "sunset glow lighting", "morning mist lighting"
        ]

    def find_images(self):
        """查找所有图像文件"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(os.path.join(self.data_dir, ext)))
            image_paths.extend(glob(os.path.join(self.data_dir, ext.upper())))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """获取单个训练样本"""
        # 1. 获取图像路径
        img_path = self.image_paths[idx]

        try:
            # 2. 加载图像
            image = PIL.Image.open(img_path).convert('RGB')

            # 3. 随机裁剪和调整大小
            image = self.random_crop_and_resize(image)

            # 4. 应用数据增强
            image = self.transform(image)

            # 5. 转换为张量并归一化到 [-1, 1]
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1) * 2.0 - 1.0

            # 6. 创建多样化的训练对
            appearance, degradation, background, mask = self.create_training_pair(image_tensor)

            # 7. 确保所有张量形状正确

            # 7.1 确保图像是3维的 [3, H, W]
            if appearance.dim() == 4:  # 如果是 [1, 3, H, W]
                appearance = appearance.squeeze(0)
            if degradation.dim() == 4:
                degradation = degradation.squeeze(0)
            if background.dim() == 4:
                background = background.squeeze(0)

            # 7.2 确保mask是3维的 [1, H, W]
            if mask.dim() == 2:  # 如果是 [H, W]
                mask = mask.unsqueeze(0)  # 变成 [1, H, W]
            elif mask.dim() == 3 and mask.shape[0] != 1:
                # 如果mask是 [C, H, W] 且 C != 1
                if mask.shape[0] == 3:  # 如果是RGB mask
                    mask = mask.mean(dim=0, keepdim=True)  # 转为灰度
                else:
                    mask = mask[:1]  # 只取第一个通道

            # 7.3 确保mask值在[0, 1]范围内
            mask = torch.clamp(mask, 0, 1)

            # 8. 创建环境光照图
            environment_map = self.create_environment_map(image_tensor)

            # 8.1 确保environment_map是3维的 [32, 32, 3]
            if environment_map.dim() == 2:  # 如果是 [32, 32]
                environment_map = environment_map.unsqueeze(-1).repeat(1, 1, 3)  # 变成 [32, 32, 3]
            elif environment_map.dim() == 4:  # 如果是 [1, 32, 32, 3]
                environment_map = environment_map.squeeze(0)  # 变成 [32, 32, 3]

            # 8.2 确保环境图值在[0, 1]范围内
            environment_map = torch.clamp(environment_map, 0, 1)

            # 9. 随机选择光照提示词
            prompt = random.choice(self.lighting_prompts)

            # 10. 验证输出形状（调试用，只在前几个样本显示）
            if idx < 3:
                print(f"\n📊 数据集样本 {idx} 形状检查:")
                print(f"  🎯 appearance:      {appearance.shape}      (应为 [3, {self.image_size}, {self.image_size}])")
                print(f"  🔧 degradation:     {degradation.shape}     (应为 [3, {self.image_size}, {self.image_size}])")
                print(
                    f"  🏞️  background:      {background.shape}      (应为 [3, {self.image_size}, {self.image_size}])")
                print(f"  🎭 mask:            {mask.shape}            (应为 [1, {self.image_size}, {self.image_size}])")
                print(f"  💡 environment_map: {environment_map.shape} (应为 [32, 32, 3])")
                print(f"  📝 prompt:          '{prompt}'")

                # 额外检查值范围
                print(f"  📈 值范围检查:")
                print(f"     appearance范围: [{appearance.min():.2f}, {appearance.max():.2f}] (应为 [-1, 1])")
                print(f"     mask范围: [{mask.min():.2f}, {mask.max():.2f}] (应为 [0, 1])")
                print(
                    f"     environment_map范围: [{environment_map.min():.2f}, {environment_map.max():.2f}] (应为 [0, 1])")

            # 11. 返回样本字典
            return {
                'appearance': appearance,  # 形状: [3, H, W]
                'degradation': degradation,  # 形状: [3, H, W]
                'background': background,  # 形状: [3, H, W]
                'mask': mask,  # 形状: [1, H, W]
                'environment_map': environment_map,  # 形状: [32, 32, 3]
                'prompt': prompt  # 类型: str
            }

        except Exception as e:
            # 12. 错误处理
            print(f"\n❌ 处理图像 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

            # 13. 返回默认样本
            print("🔄 返回默认样本...")
            return self.create_default_sample()

    def random_crop_and_resize(self, image):
        """随机裁剪并调整大小"""
        w, h = image.size
        # 随机缩放
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), PIL.Image.LANCZOS)

        # 随机裁剪
        if new_w > self.image_size and new_h > self.image_size:
            x = random.randint(0, new_w - self.image_size)
            y = random.randint(0, new_h - self.image_size)
            image = image.crop((x, y, x + self.image_size, y + self.image_size))
        else:
            image = image.resize((self.image_size, self.image_size), PIL.Image.LANCZOS)

        return image

    def create_training_pair(self, image):
        """创建训练对 - 增强版本"""
        # 目标外观 - 应用随机光照调整
        appearance = self.apply_lighting_adjustment(image.clone())

        # 退化版本 - 更强的变换
        degradation = self.create_degradation_version(image.clone())

        # 背景 - 多样化的背景生成
        background = self.create_diverse_background(image.clone())

        # 掩码 - 更真实的掩码生成
        mask = self.create_advanced_mask(image.clone())

        return appearance, degradation, background, mask

    def apply_lighting_adjustment(self, image):
        """应用光照调整"""
        # 随机选择一种光照调整方式
        method = random.choice(['brightness', 'contrast', 'color_temp', 'mixed'])

        if method == 'brightness':
            # 调整亮度
            brightness_factor = random.uniform(0.7, 1.3)
            image = image * brightness_factor
        elif method == 'contrast':
            # 调整对比度
            mean = image.mean()
            contrast_factor = random.uniform(0.8, 1.2)
            image = (image - mean) * contrast_factor + mean
        elif method == 'color_temp':
            # 调整色温
            if random.random() > 0.5:
                # 暖色调
                warm_filter = torch.tensor([1.2, 1.0, 0.8]).view(3, 1, 1)
            else:
                # 冷色调
                cool_filter = torch.tensor([0.8, 0.9, 1.2]).view(3, 1, 1)
            image = image * warm_filter if 'warm' in locals() else image * cool_filter
        else:  # mixed
            # 混合调整
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.9, 1.1)
            mean = image.mean()
            image = image * brightness_factor
            image = (image - mean) * contrast_factor + mean

        return torch.clamp(image, -1, 1)

    def create_degradation_version(self, image):
        """创建退化版本"""
        degradation = image.clone()

        # 更强的亮度调整
        brightness = random.uniform(0.3, 0.8)
        degradation = degradation * brightness

        # 添加噪声
        if random.random() > 0.3:
            noise_std = random.uniform(0.05, 0.15)
            degradation = degradation + torch.randn_like(degradation) * noise_std

        # 模糊
        if random.random() > 0.5:
            from torchvision.transforms.functional import gaussian_blur
            kernel_size = random.choice([11, 15, 21])
            degradation = gaussian_blur(degradation.unsqueeze(0), kernel_size=kernel_size)[0]

        return torch.clamp(degradation, -1, 1)

    def create_diverse_background(self, image):
        """创建多样化背景"""
        method = random.choice(['blur', 'color', 'texture', 'composite'])

        if method == 'blur':
            from torchvision.transforms.functional import gaussian_blur
            background = gaussian_blur(image.unsqueeze(0), kernel_size=51)[0]
        elif method == 'color':
            # 纯色背景
            bg_color = torch.rand(3, 1, 1) * 2 - 1
            background = bg_color.repeat(1, image.shape[1], image.shape[2])
        elif method == 'texture':
            # 纹理背景
            background = torch.randn_like(image) * 0.3
        else:  # composite
            # 混合背景
            from torchvision.transforms.functional import gaussian_blur
            blurred = gaussian_blur(image.unsqueeze(0), kernel_size=51)[0]
            noise = torch.randn_like(image) * 0.2
            background = blurred * 0.7 + noise * 0.3

        return torch.clamp(background, -1, 1)

    def create_advanced_mask(self, image):
        """创建高级掩码"""
        h, w = image.shape[1], image.shape[2]
        mask = torch.zeros(1, h, w)  # 直接创建3维张量

        # 随机选择掩码类型
        mask_type = random.choice(['ellipse', 'rectangle', 'irregular', 'gradient'])

        if mask_type == 'ellipse':
            # 椭圆掩码
            center_y, center_x = h // 2, w // 2
            ellipse_h = random.randint(h // 4, 3 * h // 4)
            ellipse_w = random.randint(w // 4, 3 * w // 4)
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            mask_data = ((x - center_x) ** 2 / (ellipse_w // 2) ** 2 +
                         (y - center_y) ** 2 / (ellipse_h // 2) ** 2) <= 1
            mask[0] = mask_data.float()

        elif mask_type == 'rectangle':
            # 矩形掩码
            rect_h = random.randint(h // 3, 2 * h // 3)
            rect_w = random.randint(w // 3, 2 * w // 3)
            start_y = random.randint(0, h - rect_h)
            start_x = random.randint(0, w - rect_w)
            mask[0, start_y:start_y + rect_h, start_x:start_x + rect_w] = 1

        elif mask_type == 'irregular':
            # 不规则掩码
            center_y, center_x = h // 2, w // 2
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distance = torch.sqrt((x - center_x).float() ** 2 + (y - center_y).float() ** 2)
            max_dist = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2))
            mask_data = (distance < max_dist * random.uniform(0.3, 0.6)).float()
            mask[0] = mask_data

        else:  # gradient
            # 渐变掩码
            center_y, center_x = h // 2, w // 2
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distance = torch.sqrt((x - center_x).float() ** 2 + (y - center_y).float() ** 2)
            max_dist = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2))
            mask_data = torch.exp(-distance / (max_dist * random.uniform(0.3, 0.7)))
            mask[0] = mask_data

        # 添加噪声使边缘更自然
        if random.random() > 0.3:
            noise = torch.randn(h, w) * 0.1
            mask[0] = torch.clamp(mask[0] + noise, 0, 1)

        return mask  # 形状: [1, H, W]

    def create_environment_map(self, image):
        """基于图像内容创建环境图"""
        # 计算图像的主要颜色
        avg_color = image.mean(dim=(1, 2)).cpu().numpy()

        # 创建基础环境图
        env_map = np.ones((32, 32, 3)) * avg_color.reshape(1, 1, 3)

        # 添加光照变化
        light_intensity = random.uniform(0.8, 1.2)
        light_color = np.random.uniform(0.8, 1.2, 3)
        env_map = env_map * light_intensity * light_color.reshape(1, 1, 3)

        # 添加随机变化
        env_map += np.random.normal(0, 0.1, (32, 32, 3))

        # 确保形状为 [32, 32, 3]
        env_map = np.clip(env_map, 0, 1)
        env_map_tensor = torch.from_numpy(env_map).float()

        return env_map_tensor  # 形状: [32, 32, 3]

    def create_default_sample(self):
        """创建默认样本"""
        # 确保所有张量有正确的形状
        image_tensor = torch.rand(3, self.image_size, self.image_size) * 2 - 1
        appearance = image_tensor
        degradation = image_tensor * 0.5
        background = torch.randn_like(image_tensor) * 0.3
        mask = torch.ones(1, self.image_size, self.image_size) * 0.5  # [1, H, W]
        environment_map = torch.rand(32, 32, 3)  # [32, 32, 3]
        prompt = "professional studio lighting"

        return {
            'appearance': appearance,  # [3, 512, 512]
            'degradation': degradation,  # [3, 512, 512]
            'background': background,  # [3, 512, 512]
            'mask': mask,  # [1, 512, 512]
            'environment_map': environment_map,  # [32, 32, 3]
            'prompt': prompt
        }# 3. 增强的损失函数类
class ICLightLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha  # 扩散损失权重
        self.beta = beta  # 光传输一致性损失权重
        self.gamma = gamma  # 感知损失权重

    def forward(self, noise_pred, noise_target, appearances=None, generated=None):
        """计算总损失"""
        # 基础扩散损失
        diffusion_loss = F.mse_loss(noise_pred, noise_target)

        total_loss = self.alpha * diffusion_loss

        # 光传输一致性损失（如果有多个光照条件）
        if appearances is not None and len(appearances) > 1:
            consistency_loss = self.light_transport_consistency_loss(appearances)
            total_loss += self.beta * consistency_loss

        # 感知损失（如果生成了图像）
        if generated is not None and appearances is not None:
            perceptual_loss = self.perceptual_similarity_loss(generated, appearances[0])
            total_loss += self.gamma * perceptual_loss

        return total_loss, {
            'diffusion_loss': diffusion_loss.item(),
            'total_loss': total_loss.item()
        }

    def light_transport_consistency_loss(self, appearances):
        """光传输一致性损失"""
        # 这里实现论文中的光传输一致性约束
        # 对于同一物体的不同光照外观，应该满足线性混合关系
        if len(appearances) < 2:
            return 0.0

        # 简化实现：确保外观变化平滑
        loss = 0.0
        for i in range(len(appearances) - 1):
            loss += F.l1_loss(appearances[i], appearances[i + 1])

        return loss / (len(appearances) - 1)

    def perceptual_similarity_loss(self, generated, target):
        """感知相似性损失"""
        # 简化实现：使用高级特征相似性
        return F.l1_loss(generated, target)


# 4. 增强的训练器类
class ICLightTrainer:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 启用内存优化
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        # 加载模型组件
        print("加载预训练模型组件...")

        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=torch.float16
        ).to(self.device)

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        # 冻结VAE和文本编码器
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # 环境编码器 - 现在这个类已经定义在前面了
        self.env_encoder = EnvironmentEncoder().to(self.device)

        # 修改UNet输入层
        self.modify_unet_input()

        # 梯度检查点
        self.unet.enable_gradient_checkpointing()

        # 损失函数
        self.criterion = ICLightLoss(alpha=1.0, beta=0.1, gamma=0.01)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.env_encoder.parameters()),
            lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # 噪声调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        print("模型初始化完成")
        self.print_memory_usage()

    def print_memory_usage(self):
        """打印内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"GPU内存使用: {allocated:.2f} GB / {reserved:.2f} GB")

    def modify_unet_input(self):
        """修改UNet输入层"""
        original_conv = self.unet.conv_in
        new_conv = nn.Conv2d(13, original_conv.out_channels, kernel_size=3, padding=1).to(self.device)

        with torch.no_grad():
            new_conv.weight[:, :4] = original_conv.weight
            new_conv.bias = original_conv.bias

        self.unet.conv_in = new_conv
        print(f"修改UNet输入通道: 4 -> 13")

    def encode_images(self, images):
        """编码图像到潜在空间"""
        if images.dim() == 3:
            images = images.unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                latents = self.vae.encode(images.half()).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        return latents.float()

    def prepare_extra_conditions(self, degradation, background, mask):
        """准备额外条件"""
        with torch.no_grad():
            # 编码退化图像和背景图像
            degradation_latent = self.encode_images(degradation)  # [batch, 4, 64, 64]
            background_latent = self.encode_images(background)  # [batch, 4, 64, 64]

            # 确保mask是4维的 [batch, 1, H, W]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 从 [batch, H, W] 变成 [batch, 1, H, W]

            # 调整mask大小到64×64
            mask_resized = F.interpolate(
                mask,  # 现在形状是 [batch, 1, H, W]
                size=degradation_latent.shape[-2:],  # (64, 64)
                mode='bilinear',  # 双线性插值
                align_corners=False
            )

            # 拼接所有条件
            extra_conditions = torch.cat([
                degradation_latent,  # [batch, 4, 64, 64]
                background_latent,  # [batch, 4, 64, 64]
                mask_resized  # [batch, 1, 64, 64]
            ], dim=1)  # 总共 4+4+1 = 9个通道

            print(f"额外条件形状: {extra_conditions.shape}")  # 调试用
            return extra_conditions  # [batch, 9, 64, 64]

    def train_step(self, batch):
        """训练步骤"""
        self.unet.train()
        self.env_encoder.train()

        torch.cuda.empty_cache()

        try:
            # 准备数据
            appearance = batch['appearance'].to(self.device)
            degradation = batch['degradation'].to(self.device)
            background = batch['background'].to(self.device)
            mask = batch['mask'].to(self.device)
            environment_map = batch['environment_map'].to(self.device)
            prompts = batch['prompt']

            batch_size = appearance.shape[0]

            # 确保mask是4维的 [batch, 1, H, W]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 添加通道维度
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

            # 调试：打印所有输入形状
            print(f"\n=== 训练步骤输入形状 ===")
            print(f"appearance: {appearance.shape}")
            print(f"degradation: {degradation.shape}")
            print(f"background: {background.shape}")
            print(f"mask: {mask.shape}")
            print(f"environment_map: {environment_map.shape}")

            # 编码目标图像
            with torch.no_grad():
                target_latents = self.encode_images(appearance)
                print(f"target_latents: {target_latents.shape}")

            # 准备额外条件
            extra_conditions = self.prepare_extra_conditions(degradation, background, mask)

            # 编码环境光照
            env_embeddings = self.env_encoder(environment_map)
            print(f"env_embeddings: {env_embeddings.shape}")

            # 编码文本提示
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                )
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
                print(f"text_embeddings: {text_embeddings.shape}")

            # 结合嵌入
            combined_embeddings = text_embeddings.clone()
            combined_embeddings[:, :3] = env_embeddings
            print(f"combined_embeddings: {combined_embeddings.shape}")

            # 添加噪声
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()

            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
            print(f"noisy_latents: {noisy_latents.shape}")

            # 准备UNet输入
            unet_input = torch.cat([noisy_latents, extra_conditions], dim=1)
            print(f"unet_input: {unet_input.shape}")  # 应该是 [batch, 13, 64, 64]

            # 预测噪声
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise_pred = self.unet(
                    unet_input, timesteps, encoder_hidden_states=combined_embeddings
                ).sample

                print(f"noise_pred: {noise_pred.shape}")

                # 计算损失
                total_loss, loss_dict = self.criterion(
                    noise_pred, noise,
                    appearances=[appearance]
                )

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.env_encoder.parameters(), 1.0)

            self.optimizer.step()

            return total_loss.item(), loss_dict

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU内存不足，跳过该批次")
                torch.cuda.empty_cache()
                return 0.0, {}
            else:
                print(f"训练步骤出错: {e}")
                import traceback
                traceback.print_exc()
                return 0.0, {}

    def train(self, dataloader, num_epochs=10, save_interval=2):
        """训练循环"""
        print("开始训练...")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            loss_history = {'diffusion_loss': 0.0, 'total_loss': 0.0}

            # 进度条
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs}')

            for batch_idx, batch in enumerate(pbar):
                loss, loss_dict = self.train_step(batch)

                if loss > 0:
                    epoch_loss += loss
                    num_batches += 1

                    # 更新损失历史
                    for k, v in loss_dict.items():
                        if k in loss_history:
                            loss_history[k] += v

                    # 更新进度条
                    if num_batches > 0:
                        avg_loss = epoch_loss / num_batches
                        pbar.set_postfix({
                            'Loss': f'{avg_loss:.4f}',
                            'Diff': f'{loss_history["diffusion_loss"] / num_batches:.4f}'
                        })

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f'Epoch {epoch} 完成. 平均损失: {avg_loss:.4f}')

                # 更新学习率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'当前学习率: {current_lr:.2e}')

                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(epoch, avg_loss, "best")
                    print(f'新的最佳模型已保存，损失: {best_loss:.4f}')

                # 定期保存检查点
                if epoch % save_interval == 0:
                    self.save_checkpoint(epoch, avg_loss)

        # 保存最终模型
        self.save_checkpoint(num_epochs, avg_loss, "final")
        print("训练完成！")

    def save_checkpoint(self, epoch, loss, prefix="checkpoint"):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'unet_state_dict': self.unet.state_dict(),
            'env_encoder_state_dict': self.env_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        filename = f"{prefix}_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        print(f"检查点已保存: {filename}")


# 5. 主训练函数
def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 检查训练图像
    data_dir = "./training_images"
    if not os.path.exists(data_dir):
        print(f"创建训练图像目录: {data_dir}")
        os.makedirs(data_dir)
        print(f"请将训练图像放入 {data_dir} 目录，然后重新运行")
        return

    image_files = glob(os.path.join(data_dir, "*.*"))
    if not image_files:
        print(f"在 {data_dir} 中没有找到图像文件！")
        return

    print(f"找到 {len(image_files)} 个训练图像")

    # 创建数据集和数据加载器
    dataset = ICLightDataset(data_dir, max_samples=min(300, len(image_files)))
    dataloader = DataLoader(
        dataset,
        batch_size=10,  # 根据GPU内存调整
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print(f"数据集大小: {len(dataset)}")

    # 初始化训练器
    trainer = ICLightTrainer()

    # 开始训练
    trainer.train(dataloader, num_epochs=10, save_interval=2)

if __name__ == "__main__":
    main()