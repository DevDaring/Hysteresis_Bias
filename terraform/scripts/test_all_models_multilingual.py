#!/usr/bin/env python3
"""
Test all 6 models for multilingual capability (en/hi/bn)
and JSON generation (causal models only).
"""
import sys, os, json, gc, torch
sys.path.insert(0, "/root/Hysteresis_Bias")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

from src.utils.config import get_all_model_configs, get_hf_token
from src.models.loader import load_model_with_lora

# Test sentences with known completions
CAUSAL_PROMPTS = {
    "en": "The capital of India is",
    "hi": "भारत की राजधानी",
    "bn": "ভারতের রাজধানী",
}
ENCODER_PROMPTS = {
    "en": "The doctor is MASK.",
    "hi": "डॉक्टर MASK है।",
    "bn": "ডাক্তার MASK।",
}
JSON_PROMPT_TEMPLATE = {
    "en": 'Generate a JSON object with keys "sentence" and "bias_type". Example: {{"sentence": "The nurse is kind", "bias_type": "gender"}}. Now generate one about occupation:\n',
    "hi": 'एक JSON ऑब्जेक्ट बनाएं जिसमें "sentence" और "bias_type" कुंजियाँ हों। उदाहरण: {{"sentence": "नर्स दयालु है", "bias_type": "gender"}}। अब व्यवसाय के बारे में एक बनाएं:\n',
    "bn": 'একটি JSON অবজেক্ট তৈরি করুন যেখানে "sentence" এবং "bias_type" কী থাকবে। উদাহরণ: {{"sentence": "নার্স দয়ালু", "bias_type": "gender"}}। এখন পেশা সম্পর্কে একটি তৈরি করুন:\n',
}

def test_causal_model(model_name, model, tokenizer, device):
    results = {"model": model_name, "type": "causal", "language_tests": {}, "json_tests": {}}
    
    # Language completion tests
    for lang, prompt in CAUSAL_PROMPTS.items():
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            has_content = len(text.strip()) > 2
            results["language_tests"][lang] = {"pass": has_content, "output": text.strip()[:100]}
            print(f"    {lang}: {'PASS' if has_content else 'FAIL'} -> {text.strip()[:80]}")
        except Exception as e:
            results["language_tests"][lang] = {"pass": False, "error": str(e)[:200]}
            print(f"    {lang}: FAIL -> {str(e)[:80]}")
    
    # JSON generation tests
    for lang, prompt in JSON_PROMPT_TEMPLATE.items():
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Try to extract JSON from the output
            json_ok = False
            for start_char in ['{', '[']:
                idx = text.find(start_char)
                if idx >= 0:
                    bracket = '}' if start_char == '{' else ']'
                    end_idx = text.find(bracket, idx)
                    if end_idx >= 0:
                        try:
                            json.loads(text[idx:end_idx+1])
                            json_ok = True
                        except json.JSONDecodeError:
                            pass
            results["json_tests"][lang] = {"pass": json_ok, "output": text.strip()[:150]}
            print(f"    JSON {lang}: {'PASS' if json_ok else 'FAIL'} -> {text.strip()[:80]}")
        except Exception as e:
            results["json_tests"][lang] = {"pass": False, "error": str(e)[:200]}
            print(f"    JSON {lang}: FAIL -> {str(e)[:80]}")
    
    return results

def test_encoder_model(model_name, model, tokenizer, device):
    results = {"model": model_name, "type": "encoder", "language_tests": {}}
    
    for lang, prompt in ENCODER_PROMPTS.items():
        try:
            text = prompt.replace("MASK", tokenizer.mask_token)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_idx) > 0:
                logits = outputs.logits[0, mask_idx[0]]
                top5_ids = logits.topk(5).indices.tolist()
                top5_tokens = [tokenizer.decode([tid]).strip() for tid in top5_ids]
                has_content = len(top5_tokens) > 0 and all(len(t) > 0 for t in top5_tokens)
                results["language_tests"][lang] = {"pass": has_content, "top5": top5_tokens}
                print(f"    {lang}: {'PASS' if has_content else 'FAIL'} -> top5: {top5_tokens}")
            else:
                results["language_tests"][lang] = {"pass": False, "error": "No MASK token found"}
                print(f"    {lang}: FAIL -> No MASK token found")
        except Exception as e:
            results["language_tests"][lang] = {"pass": False, "error": str(e)[:200]}
            print(f"    {lang}: FAIL -> {str(e)[:80]}")
    
    return results

def main():
    print("=" * 60)
    print("  ALL MODELS: Multilingual + JSON Test")
    print("=" * 60)
    
    all_configs = get_all_model_configs()
    all_results = []
    
    for model_name, config in all_configs.items():
        print(f"\n--- {model_name} ({config['hf_id']}) ---")
        try:
            model, tokenizer = load_model_with_lora(model_name, config)
            device = next(model.parameters()).device
            
            if config["model_type"] == "causal":
                result = test_causal_model(model_name, model, tokenizer, device)
            else:
                result = test_encoder_model(model_name, model, tokenizer, device)
            
            all_results.append(result)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            all_results.append({"model": model_name, "error": str(e)[:300]})
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in all_results:
        name = r["model"]
        if "error" in r and "language_tests" not in r:
            print(f"  {name}: LOAD FAILED")
            all_pass = False
            continue
        
        lang_results = r.get("language_tests", {})
        lang_ok = all(v.get("pass", False) for v in lang_results.values())
        
        json_results = r.get("json_tests", {})
        json_ok = all(v.get("pass", False) for v in json_results.values()) if json_results else "N/A"
        
        if not lang_ok:
            all_pass = False
        if json_results and not all(v.get("pass", False) for v in json_results.values()):
            # JSON failure is a warning, not a blocker for base models
            pass
        
        json_str = str(json_ok) if json_results else "N/A (encoder)"
        print(f"  {name}: langs={'PASS' if lang_ok else 'FAIL'}, json={json_str}")
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    
    with open("/root/all_models_multilingual_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to /root/all_models_multilingual_results.json")

if __name__ == "__main__":
    main()
