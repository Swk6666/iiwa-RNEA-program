#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Pinocchio加载KUKA IIWA 14机器人模型并输出每个link的6维惯量矩阵

作者: Auto-generated
日期: 2025-12-27
"""

import numpy as np
import pinocchio as pin
import os

def print_inertia_matrices(xml_path):
    """
    加载机器人模型并打印每个link的6维惯量矩阵
    
    参数:
        xml_path: MuJoCo XML文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(xml_path):
        print(f"错误: 文件 {xml_path} 不存在!")
        return
    
    print(f"正在从 {xml_path} 加载机器人模型...")
    print("="*80)
    
    try:
        # 使用Pinocchio加载URDF文件
        model = pin.buildModelFromUrdf(xml_path)
        
        print(f"\n成功加载模型: {model.name}")
        print(f"关节数量: {model.njoints}")
        print(f"自由度数量: {model.nv}")
        print(f"配置空间维度: {model.nq}")
        print("="*80)
        
        # 遍历所有的link(body)并输出其惯量信息
        for i, inertia in enumerate(model.inertias):
            # 获取link名称
            link_name = model.names[i]
            
            print(f"\n--- Link {i}: {link_name} ---")
            print(f"质量: {inertia.mass:.6f} kg")
            
            # 获取质心位置(在link坐标系中)
            com = inertia.lever
            print(f"质心位置 (相对于link坐标系):")
            print(f"  x: {com[0]:.6f} m")
            print(f"  y: {com[1]:.6f} m")
            print(f"  z: {com[2]:.6f} m")
            
            # 获取6x6惯量矩阵(空间惯量矩阵)
            # 这是完整的6维惯量矩阵,包含质量、质心和转动惯量信息
            spatial_inertia = inertia.matrix()
            
            print(f"\n6x6 空间惯量矩阵 (Spatial Inertia Matrix):")
            print("格式: [[角动量部分], [线性动量部分]]")
            print("     上半部分(3x6): 与角速度相关的惯量")
            print("     下半部分(3x6): 与线速度相关的惯量")
            print("-"*60)
            
            # 打印矩阵,保留更多小数位以提高精度
            for row in spatial_inertia:
                print("  [", end="")
                for j, val in enumerate(row):
                    if j == len(row) - 1:
                        print(f"{val:12.6f}", end="")
                    else:
                        print(f"{val:12.6f}, ", end="")
                print("]")
            
            # 单独输出转动惯量部分(相对于质心的转动惯量)
            print(f"\n3x3 转动惯量矩阵 (相对于质心):")
            rotational_inertia = inertia.inertia
            for row in rotational_inertia:
                print("  [", end="")
                for j, val in enumerate(row):
                    if j == len(row) - 1:
                        print(f"{val:12.6f}", end="")
                    else:
                        print(f"{val:12.6f}, ", end="")
                print("]")
            
            print("-"*80)
        
        print("\n所有link的惯量信息已输出完毕!")
        
        # 额外信息: 输出关节信息
        print("\n" + "="*80)
        print("关节信息:")
        print("="*80)
        for i in range(1, model.njoints):  # 从1开始,0是universe
            joint_name = model.names[i]
            joint_id = i
            parent_id = model.parents[i]
            parent_name = model.names[parent_id]
            
            print(f"关节 {i}: {joint_name}")
            print(f"  父link: {parent_name} (id: {parent_id})")
            print(f"  关节类型: {model.joints[i]}")
            print()
        
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_inertia_matrices_to_file(xml_path, output_file="inertia_matrices.txt"):
    """
    加载机器人模型并将每个link的6维惯量矩阵保存到文件
    
    参数:
        xml_path: MuJoCo XML文件的路径
        output_file: 输出文件名
    """
    if not os.path.exists(xml_path):
        print(f"错误: 文件 {xml_path} 不存在!")
        return
    
    try:
        model = pin.buildModelFromUrdf(xml_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"机器人模型: {model.name}\n")
            f.write(f"关节数量: {model.njoints}\n")
            f.write(f"自由度数量: {model.nv}\n")
            f.write("="*80 + "\n\n")
            
            for i, inertia in enumerate(model.inertias):
                link_name = model.names[i]
                
                f.write(f"Link {i}: {link_name}\n")
                f.write(f"质量: {inertia.mass:.6f} kg\n")
                
                com = inertia.lever
                f.write(f"质心位置: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] m\n\n")
                
                spatial_inertia = inertia.matrix()
                f.write("6x6 空间惯量矩阵:\n")
                np.savetxt(f, spatial_inertia, fmt='%12.6f')
                
                f.write("\n3x3 转动惯量矩阵 (相对于质心):\n")
                rotational_inertia = inertia.inertia
                np.savetxt(f, rotational_inertia, fmt='%12.6f')
                
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"惯量矩阵已保存到文件: {output_file}")
        return model
        
    except Exception as e:
        print(f"保存惯量矩阵时出错: {e}")
        return None


if __name__ == "__main__":
    # XML文件路径
    xml_path = "iiwa_description/urdf/iiwa14.urdf"
    
    # 打印所有惯量矩阵到控制台
    print("=" * 80)
    print(" KUKA IIWA 14 机器人 - 惯量矩阵输出")
    print("=" * 80)
    model = print_inertia_matrices(xml_path)
    
    # 同时保存到文件
    if model is not None:
        print("\n" + "="*80)
        save_inertia_matrices_to_file(xml_path, "iiwa14_inertia_matrices.txt")
        print("="*80)

