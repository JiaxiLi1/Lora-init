#!/usr/bin/env python3
"""
简单测试脚本，验证LORO-main中的T5支持
"""

import torch
from transformers import AutoTokenizer
from transformers.models.t5.configuration_t5 import T5Config
from peft_pretraining.modeling_t5 import T5ForConditionalGeneration
from t5_data_utils import create_t5_denoising_batch

def test_t5_model_loading():
    """测试T5模型加载"""
    print("🔧 Testing T5 model loading...")
    
    try:
        # 加载T5-small配置
        config = T5Config.from_pretrained('configs/t5_small_normal.json')
        
        # 创建T5模型
        model = T5ForConditionalGeneration(config)
        print(f'✅ T5 model created successfully with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters')
        
        return model
    except Exception as e:
        print(f"❌ T5 model loading failed: {e}")
        return None

def test_t5_data_processing():
    """测试T5数据处理"""
    print("🔧 Testing T5 data processing...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        
        # 创建样本batch
        sample_text = 'Today is a beautiful day and we should go to the park to play'
        tokenized = tokenizer(sample_text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        
        batch = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        
        # 处理为T5去噪batch
        t5_batch = create_t5_denoising_batch(batch, tokenizer, max_length=64)
        
        print(f'✅ T5 data processing successful!')
        print(f'   Original text: {sample_text}')
        print(f'   Encoder input: {tokenizer.decode(t5_batch["input_ids"][0], skip_special_tokens=False)}')
        print(f'   Decoder input: {tokenizer.decode(t5_batch["decoder_input_ids"][0], skip_special_tokens=False)}')
        
        return tokenizer, t5_batch
    except Exception as e:
        print(f"❌ T5 data processing failed: {e}")
        return None, None

def test_t5_forward_pass():
    """测试T5前向传播"""
    print("🔧 Testing T5 forward pass...")
    
    model = test_t5_model_loading()
    tokenizer, t5_batch = test_t5_data_processing()
    
    if model is None or tokenizer is None or t5_batch is None:
        print("❌ Cannot test forward pass due to previous failures")
        return False
    
    try:
        with torch.no_grad():
            output = model(
                input_ids=t5_batch["input_ids"],
                attention_mask=t5_batch["attention_mask"],
                decoder_input_ids=t5_batch["decoder_input_ids"],
                decoder_attention_mask=t5_batch["decoder_attention_mask"],
                labels=t5_batch["labels"]
            )
        
        print(f'✅ T5 forward pass successful! Loss: {output.loss.item():.4f}')
        return True
    except Exception as e:
        print(f"❌ T5 forward pass failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting T5 LORO integration tests...")
    print("=" * 60)
    
    # 运行所有测试
    success = test_t5_forward_pass()
    
    print("=" * 60)
    if success:
        print("🎉 All T5 LORO tests passed!")
        print("✅ You can now use --model_type t5 with --optimizer loro_adamw")
    else:
        print("❌ Some T5 LORO tests failed!")
        print("🔧 Please check the error messages above")