import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from glob import glob
import random


class RealDataPreparer:
    def __init__(self, data_dir="./training_images"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def find_images(self, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """查找数据目录中的所有图像"""
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(os.path.join(self.data_dir, f"*{ext}")))
            image_paths.extend(glob(os.path.join(self.data_dir, f"*{ext.upper()}")))
        return image_paths

    def create_training_pairs(self):
        """为每张图像创建训练对"""
        image_paths = self.find_images()

        if not image_paths:
            print(f"在 {self.data_dir} 中没有找到图像文件！")
            print("请将训练图像放入该目录，支持格式: jpg, jpeg, png, bmp")
            return []

        print(f"找到 {len(image_paths)} 张图像")

        training_pairs = []

        for i, img_path in enumerate(image_paths):
            print(f"处理图像 {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

            try:
                # 加载原始图像
                image = Image.open(img_path).convert('RGB')
                if image.size[0] < 256 or image.size[1] < 256:
                    print(f"跳过尺寸太小的图像: {img_path}")
                    continue

                image_tensor = self.transform(image)

                # 为每张图像创建多个训练样本
                for j in range(3):  # 每张图像创建3个样本
                    training_pair = self.create_single_training_pair(
                        image_tensor, img_path, j
                    )
                    if training_pair:
                        training_pairs.append(training_pair)

            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue

        return training_pairs

    def create_single_training_pair(self, original_image, original_path, variant_id):
        """为单张图像创建训练对"""
        try:
            # 创建不同的光照版本作为目标
            target_image = self.create_variant_illumination(original_image)

            # 创建退化版本作为输入
            degradation_image = self.create_degradation_version(original_image)

            # 创建背景（使用模糊版本或纯色）
            background_image = self.create_background(original_image)

            # 创建掩码（基于简单分割或中心区域）
            mask = self.create_simple_mask(original_image)

            # 创建环境图
            env_map = self.estimate_environment_map(target_image)

            # 生成提示词
            prompt = self.generate_lighting_prompt()

            # 保存临时文件
            base_name = f"{os.path.splitext(os.path.basename(original_path))[0]}_v{variant_id}"

            return {
                'appearance_path': self.save_temp_image(target_image, f"{base_name}_target"),
                'degradation_path': self.save_temp_image(degradation_image, f"{base_name}_degradation"),
                'background_path': self.save_temp_image(background_image, f"{base_name}_background"),
                'env_map_path': self.save_env_map(env_map, base_name),
                'mask_path': self.save_temp_image(mask, f"{base_name}_mask", grayscale=True),
                'prompt': prompt
            }

        except Exception as e:
            print(f"创建训练对时出错: {e}")
            return None

    def create_variant_illumination(self, image):
        """创建不同的光照版本"""
        # 随机选择一种光照变换
        methods = [
            self.adjust_brightness_contrast,
            self.adjust_color_temperature,
            self.add_light_effects
        ]
        method = random.choice(methods)
        return method(image)

    def adjust_brightness_contrast(self, image):
        """调整亮度和对比度"""
        brightness = 0.6 + 0.8 * random.random()  # 0.6-1.4
        contrast = 0.7 + 0.6 * random.random()  # 0.7-1.3

        # 调整亮度
        result = image * brightness
        result = torch.clamp(result, 0, 1)

        # 调整对比度
        mean = result.mean()
        result = (result - mean) * contrast + mean
        return torch.clamp(result, 0, 1)

    def adjust_color_temperature(self, image):
        """调整色温"""
        # 暖色调或冷色调
        if random.random() > 0.5:
            # 暖色调 (增加红色/黄色)
            warm_filter = torch.tensor([1.2, 1.0, 0.8]).view(3, 1, 1)
            result = image * warm_filter
        else:
            # 冷色调 (增加蓝色)
            cool_filter = torch.tensor([0.8, 0.9, 1.2]).view(3, 1, 1)
            result = image * cool_filter

        return torch.clamp(result, 0, 1)

    def add_light_effects(self, image):
        """添加光照效果"""
        result = image.clone()
        h, w = image.shape[1], image.shape[2]

        # 创建光源效果
        light_center_x = random.randint(w // 4, 3 * w // 4)
        light_center_y = random.randint(h // 4, 3 * h // 4)
        light_radius = random.randint(50, 200)

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        distance = torch.sqrt((x - light_center_x) ** 2 + (y - light_center_y) ** 2)
        light_mask = torch.exp(-distance / light_radius).unsqueeze(0)

        # 应用光照
        light_strength = 0.3 + 0.4 * random.random()
        result = result + light_mask * light_strength

        return torch.clamp(result, 0, 1)

    def create_degradation_version(self, image):
        """创建退化版本"""
        # 应用更强的变换作为输入条件
        degradation = image.clone()

        # 随机调整
        degradation = degradation * (0.4 + 0.4 * random.random())  # 亮度
        degradation = torch.clamp(degradation + 0.1 * torch.randn_like(degradation), 0, 1)  # 噪声

        return degradation

    def create_background(self, image):
        """创建背景"""
        # 方法1: 使用模糊版本
        from torchvision.transforms.functional import gaussian_blur
        background = gaussian_blur(image, kernel_size=31)

        # 方法2: 随机纯色背景
        if random.random() > 0.7:
            bg_color = torch.rand(3, 1, 1)
            background = bg_color.repeat(1, image.shape[1], image.shape[2])

        return background

    def create_simple_mask(self, image):
        """创建简单掩码"""
        h, w = image.shape[1], image.shape[2]
        mask = torch.zeros(1, h, w)

        # 创建中心椭圆掩码
        center_y, center_x = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))

        # 随机椭圆大小
        ellipse_h = random.randint(h // 3, 2 * h // 3)
        ellipse_w = random.randint(w // 3, 2 * w // 3)

        mask = ((x - center_x) ** 2 / (ellipse_w // 2) ** 2 +
                (y - center_y) ** 2 / (ellipse_h // 2) ** 2) <= 1

        # 添加一些随机性
        mask = mask.float()
        if random.random() > 0.5:
            # 添加噪声使边缘更自然
            noise = torch.randn(h, w) * 0.1
            mask = torch.clamp(mask + noise.unsqueeze(0), 0, 1)

        return mask

    def estimate_environment_map(self, image):
        """估计环境光照图"""
        # 简化实现 - 实际应该使用更复杂的方法
        # 这里基于图像颜色统计创建环境图
        avg_color = image.mean(dim=(1, 2)).numpy()
        env_map = np.ones((32, 32, 3)) * avg_color.reshape(1, 1, 3)

        # 添加一些变化
        env_map += np.random.normal(0, 0.1, (32, 32, 3))
        return np.clip(env_map, 0, 1)

    def generate_lighting_prompt(self):
        """生成光照提示词"""
        lighting_types = [
            "studio lighting", "natural sunlight", "sunset glow",
            "morning light", "golden hour", "blue hour",
            "overcast daylight", "dramatic lighting", "soft light",
            "hard light", "warm lighting", "cool lighting"
        ]

        adjectives = [
            "beautiful", "professional", "dramatic", "soft", "harsh",
            "warm", "cool", "natural", "artificial", "cinematic"
        ]

        lighting = random.choice(lighting_types)
        adjective = random.choice(adjectives)

        return f"{adjective} {lighting}"

    def save_temp_image(self, tensor, name, grayscale=False):
        """保存临时图像"""
        os.makedirs("temp_images", exist_ok=True)
        path = os.path.join("temp_images", f"{name}.jpg")

        if tensor.dim() == 3:
            if grayscale:
                # 灰度图像
                img_np = (tensor[0].numpy() * 255).astype(np.uint8)
            else:
                # RGB图像
                img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            Image.fromarray(img_np).save(path)
        return path

    def save_env_map(self, env_map, name):
        """保存环境图"""
        os.makedirs("temp_env_maps", exist_ok=True)
        path = os.path.join("temp_env_maps", f"{name}.npy")
        np.save(path, env_map)
        return path


def prepare_real_training_data():
    """准备真实训练数据"""

    print("开始准备训练数据...")

    preparer = RealDataPreparer("./training_images")
    training_data = preparer.create_training_pairs()

    if not training_data:
        print("没有生成任何训练数据！")
        print("请执行以下步骤：")
        print("1. 创建 'training_images' 文件夹")
        print("2. 将您的训练图像放入该文件夹")
        print("3. 重新运行此脚本")
        return

    # 保存训练数据列表
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"成功准备了 {len(training_data)} 个训练样本")
    print("训练数据已保存到: training_data.json")
    print("临时文件保存在: temp_images/ 和 temp_env_maps/")

    return training_data


if __name__ == "__main__":
    prepare_real_training_data()