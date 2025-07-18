import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np
import os


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for HRNet."""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """High-resolution module with parallel branches and multi-scale fusion."""
    
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, 
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
    
    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)
        
        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)
        
        if num_branches != len(num_inchannels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(num_inchannels)})'
            raise ValueError(error_msg)
    
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )
        
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        
        return nn.Sequential(*layers)
    
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i])
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)
    
    def get_num_inchannels(self):
        return self.num_inchannels
    
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse


class HRNet(nn.Module):
    """
    HRNet-W32 implementation for cephalometric landmark detection.
    """
    
    def __init__(self, config):
        super(HRNet, self).__init__()
        
        # Parse configuration
        self.config = config
        self.num_joints = config.get('NUM_JOINTS', 19)
        
        # Define blocks
        blocks_dict = {
            'BASIC': BasicBlock,
            'BOTTLENECK': Bottleneck
        }
        
        # Stage 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1 residual layers
        self.layer1 = self._make_layer(
            Bottleneck, 64, 64, 4
        )
        
        # Stage 2
        stage2_cfg = config['STAGE2']
        num_channels = stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(stage2_cfg, num_channels)
        
        # Stage 3
        stage3_cfg = config['STAGE3']
        num_channels = stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(stage3_cfg, num_channels)
        
        # Stage 4
        stage4_cfg = config['STAGE4']
        num_channels = stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(stage4_cfg, num_channels, multi_scale_output=True)
        
        # Final layer for landmark prediction
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=self.num_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = BasicBlock if layer_config['BLOCK'] == 'BASIC' else Bottleneck
        fuse_method = layer_config.get('FUSE_METHOD', 'SUM')
        
        modules = []
        for i in range(num_modules):
            # Multi-scale output is only used for the last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        
        return nn.Sequential(*modules), num_inchannels
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        # Stage 2
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Stage 3
        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Stage 4
        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Only use the highest resolution feature map
        x = y_list[0]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final prediction layer
        x = self.final_layer(x)
        
        # Upsample to match target heatmap size 
        # Don't upsample - keep at the natural resolution from the highest-resolution branch
        # The target heatmaps are generated at 192x192 to match this
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 3, 768, 768)):
        # Rough FLOPS estimation
        # This is a simplified calculation
        dummy_input = torch.randn(input_size)
        flops = 0
        
        # Conv1: 3*768*768 * 64*3*3 * 1 / 4 (stride=2)
        flops += 3 * 768 * 768 * 64 * 3 * 3 / 4
        
        # Conv2: 64*384*384 * 64*3*3 * 1 / 4 (stride=2)
        flops += 64 * 384 * 384 * 64 * 3 * 3 / 4
        
        # Rough approximation for the rest
        flops += 4.5e9  # Approximately 4.5 GFLOPs for HRNet-W32
        
        return flops


def get_hrnet_w32(config: Dict[str, Any]) -> HRNet:
    """
    Create HRNet-W32 model.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        HRNet model
    """
    return HRNet(config)


def load_pretrained_hrnet(model: HRNet, pretrained_path: str) -> HRNet:
    """
    Load pretrained weights for HRNet.
    
    Args:
        model: HRNet model
        pretrained_path: Path to pretrained weights
    
    Returns:
        Model with loaded weights
    """
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded successfully")
    else:
        print("No pretrained weights found, training from scratch")
    
    return model


def test_hrnet_model():
    """Test HRNet model creation and forward pass."""
    # Create model configuration
    config = {
        'NUM_JOINTS': 19,
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [32, 64],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128, 256],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM'
        }
    }
    
    # Create model
    model = get_hrnet_w32(config)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 768, 768)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 19, 768, 768)")
    
    # Check if output shape is correct
    assert output.shape == (batch_size, 19, 768, 768), f"Wrong output shape: {output.shape}"
    print("Model test passed!")


if __name__ == "__main__":
    test_hrnet_model() 