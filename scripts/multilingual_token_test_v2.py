#!/usr/bin/env python3
"""
多语言 Token 消耗对比测试脚本 v2

严格控制：
1. 所有语言输入 token 数相同
2. 所有语言语义完全一致
3. 生成内容有足够复杂度
4. 重复 15 次测试取统计

测试模型：
1. gemma3:1b
2. carstenuhlig/omnicoder-9b:latest
"""

import subprocess
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import statistics


# 标准化测试任务（所有语言语义完全相同）
# 任务设计：中等难度，有标准答案，可量化评估
TEST_TASKS = {
    "math": {
        "name": "数学推理",
        "difficulty": "medium",
        "expected_tokens": 150
    },
    "logic": {
        "name": "逻辑推理", 
        "difficulty": "medium",
        "expected_tokens": 120
    },
    "knowledge": {
        "name": "知识问答",
        "difficulty": "medium",
        "expected_tokens": 100
    }
}


# 多语言 Prompt（严格控制语义一致）
# 每个任务都有 11 种语言版本，语义完全相同
MULTILINGUAL_PROMPTS = {
    "math": {
        "en": "If a train travels 120 kilometers in 2 hours, and then continues at the same speed for another 3 hours, what is the total distance traveled? Please show your calculation steps and explain your reasoning.",
        "zh": "如果一列火车 2 小时行驶 120 公里，然后以相同速度继续行驶 3 小时，总行驶距离是多少？请展示计算步骤并解释推理过程。",
        "ja": "列車が 2 時間で 120 キロメートル走行し、その後同じ速度で 3 時間走行し続けた場合、総走行距離は何キロですか。計算手順と推論を説明してください。",
        "fr": "Si un train parcourt 120 kilomètres en 2 heures, puis continue à la même vitesse pendant 3 heures, quelle est la distance totale parcourue ? Montrez vos calculs et expliquez votre raisonnement.",
        "ru": "Если поезд проезжает 120 километров за 2 часа, а затем продолжает движение с той же скоростью ещё 3 часа, каково общее расстояние? Покажите вычисления и объясните рассуждения.",
        "de": "Wenn ein Zug 120 Kilometer in 2 Stunden fährt und dann mit gleicher Geschwindigkeit 3 Stunden weiterfährt, wie groß ist die Gesamtstrecke? Zeigen Sie Ihre Berechnungen und erklären Sie Ihre Überlegungen.",
        "es": "Si un tren recorre 120 kilómetros en 2 horas y luego continúa a la misma velocidad durante 3 horas, ¿cuál es la distancia total? Muestre sus cálculos y explique su razonamiento.",
        "pt": "Se um trem percorre 120 quilômetros em 2 horas e depois continua na mesma velocidade por 3 horas, qual é a distância total? Mostre seus cálculos e explique seu raciocínio.",
        "it": "Se un treno percorre 120 chilometri in 2 ore e poi continua alla stessa velocità per 3 ore, qual è la distanza totale? Mostri i calcoli e spieghi il ragionamento.",
        "ko": "기차가 2 시간 동안 120 킬로미터를 이동한 후 같은 속도로 3 시간 더 이동하면 총 이동 거리는 얼마입니까 ? 계산 단계와 추론 과정을 보여주세요.",
        "lzh": "车行二时百二十里，复以同速行三时，问总计几里？请列算式并释其理。"
    },
    "logic": {
        "en": "All cats are mammals. All mammals have hearts. Some pets are cats. Based on these premises, can we conclude that some pets have hearts? Explain your logical reasoning step by step.",
        "zh": "所有猫都是哺乳动物。所有哺乳动物都有心脏。有些宠物是猫。基于这些前提，我们能得出有些宠物有心脏的结论吗？请逐步解释你的逻辑推理过程。",
        "ja": "全ての猫は哺乳類です。全ての哺乳類は心臓を持っています。一部のペットは猫です。これらの前提に基づき、一部のペットは心臓を持つと結論付けられますか。論理的推論を段階的に説明してください。",
        "fr": "Tous les chats sont des mammifères. Tous les mammifères ont un cœur. Certains animaux de compagnie sont des chats. Peut-on conclure que certains animaux de compagnie ont un cœur ? Expliquez votre raisonnement logique étape par étape.",
        "ru": "Все кошки - млекопитающие. Все млекопитающие имеют сердце. Некоторые домашние животные - кошки. Можно ли сделать вывод, что у некоторых домашних животных есть сердце ? Объясните логические рассуждения пошагово.",
        "de": "Alle Katzen sind Säugetiere. Alle Säugetiere haben ein Herz. Einige Haustiere sind Katzen. Können wir daraus schließen, dass einige Haustiere ein Herz haben ? Erklären Sie Ihre logischen Überlegungen Schritt für Schritt.",
        "es": "Todos los gatos son mamíferos. Todos los mamíferos tienen corazón. Algunas mascotas son gatos. ¿Podemos concluir que algunas mascotas tienen corazón ? Explique su razonamiento lógico paso a paso.",
        "pt": "Todos os gatos são mamíferos. Todos os mamíferos têm coração. Alguns animais de estimação são gatos. Podemos concluir que alguns animais de estimação têm coração ? Explique seu raciocínio lógico passo a passo.",
        "it": "Tutti i gatti sono mammiferi. Tutti i mammiferi hanno un cuore. Alcuni animali domestici sono gatti. Possiamo concludere che alcuni animali domestici hanno un cuore ? Spieghi il ragionamento logico passo dopo passo.",
        "ko": "모든 고양이는 포유류입니다 . 모든 포유류는 심장을 가집니다 . 일부 반려동물은 고양이입니다 . 이 전제에 근거하여 일부 반려동물이 심장을 가진다고 결론지을 수 있습니까 ? 논리적 추론을 단계별로 설명하세요.",
        "lzh": "凡猫皆哺乳，凡哺乳皆有心，有宠物为猫。可推有宠物有心乎？请次第释其理。"
    },
    "knowledge": {
        "en": "What is the capital of France? Please provide three interesting facts about this city, including its population, one famous landmark, and one historical event that happened there.",
        "zh": "法国的首都是哪里？请提供关于这个城市的三个有趣事实，包括人口数量、一个著名地标和发生在这里的一个历史事件。",
        "ja": "フランスの首都はどこですか。この都市について 3 つの興味深い事実（人口、有名なランドマーク 1 つ、そこで起こった歴史的事件 1 つ）を教えてください。",
        "fr": "Quelle est la capitale de la France ? Donnez trois faits intéressants sur cette ville, y compris sa population, un monument célèbre et un événement historique qui s'y est produit.",
        "ru": "Какая столица у Франции ? Пожалуйста , приведите три интересных факта об этом городе : численность населения , одну известную достопримечательность и одно историческое событие.",
        "de": "Was ist die Hauptstadt von Frankreich ? Nennen Sie drei interessante Fakten über diese Stadt , einschließlich Einwohnerzahl , einem berühmten Wahrzeichen und einem historischen Ereignis.",
        "es": "¿Cuál es la capital de Francia ? Proporcione tres datos interesantes sobre esta ciudad , incluyendo su población , un monumento famoso y un evento histórico que ocurrió allí.",
        "pt": "Qual é a capital da França ? Forneça três fatos interessantes sobre esta cidade , incluindo população , um marco famoso e um evento histórico que ocorreu lá.",
        "it": "Qual è la capitale della Francia ? Fornisca tre fatti interessanti su questa città , inclusa la popolazione , un monumento famoso e un evento storico accaduto lì.",
        "ko": "프랑스의 수도는 어디입니까 ? 이 도시에 대해 세 가지 흥미로운 사실 ( 인구 , 유명한 랜드마크 하나 , 거기서 일어난 역사적 사건 하나) 을 제공해주세요.",
        "lzh": "法兰西国都何名？请述三事：民数几何、名胜其一、史事其一。"
    }
}


@dataclass
class TestResult:
    """单次测试结果"""
    model: str
    language: str
    language_name: str
    task_type: str
    run_number: int
    timestamp: str = ""
    
    # 输入指标
    input_tokens: int = 0
    input_length_chars: int = 0
    
    # 输出指标
    output_tokens: int = 0
    output_length_chars: int = 0
    total_tokens: int = 0
    
    # 性能指标
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_s: float = 0.0
    
    # 质量指标
    output_complexity: str = ""  # simple/medium/complex
    has_reasoning: bool = False
    has_calculation: bool = False
    
    # 元数据
    success: bool = False
    error: str = ""


@dataclass
class LanguageStats:
    """语言统计数据"""
    language: str
    language_name: str
    task_type: str
    
    # Token 统计
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    std_input_tokens: float = 0.0
    std_output_tokens: float = 0.0
    std_total_tokens: float = 0.0
    
    # 性能统计
    avg_speed: float = 0.0
    std_speed: float = 0.0
    avg_ttft: float = 0.0
    std_ttft: float = 0.0
    
    # 成功率
    success_rate: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0


def count_chars(text: str) -> int:
    """字符数统计（用于估算 token 数）"""
    return len(text)


def run_moxing_test(model: str, prompt: str, max_tokens: int = 512) -> Dict:
    """使用 moxing 运行单次测试"""
    cmd = [
        "moxing", "ollama", "run", model,
        "-p", prompt,
        "-n", str(max_tokens),
        "-v"
    ]
    
    try:
        start_time = time.time()
        first_token_time = None
        full_output = ""
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        total_time = time.time() - start_time
        full_output = proc.stdout + proc.stderr
        
        result = {
            "success": False,
            "full_output": full_output,
            "total_time": total_time,
            "first_token_time": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_second": 0,
            "response": ""
        }
        
        # 解析性能指标
        if "Tokens:" in full_output:
            result["success"] = True
            
            # 解析 tokens 信息
            match = re.search(r'Tokens:\s*(\d+)\s*\|\s*Time:\s*([\d.]+)\s*s\s*\|\s*Speed:\s*([\d.]+)\s*tok/s\s*\|\s*TTFT:\s*([\d.]+)\s*s', full_output)
            if match:
                result["total_tokens"] = int(match.group(1))
                result["total_time"] = float(match.group(2))
                result["tokens_per_second"] = float(match.group(3))
                result["first_token_time"] = float(match.group(4))
            
            # 解析 prompt/completion tokens
            match = re.search(r'Prompt Tokens:\s*(\d+)', full_output)
            if match:
                result["prompt_tokens"] = int(match.group(1))
            
            match = re.search(r'Completion Tokens:\s*(\d+)', full_output)
            if match:
                result["completion_tokens"] = int(match.group(1))
            
            # 提取响应内容
            if "Response:" in full_output:
                response_start = full_output.find("Response:") + 9
                response_end = full_output.find("┌─", response_start)
                if response_end == -1:
                    response_end = full_output.find("Tokens:", response_start)
                if response_end == -1:
                    response_end = len(full_output)
                result["response"] = full_output[response_start:response_end].strip()
        
        return result
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout", "full_output": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "full_output": ""}


def analyze_output_quality(response: str, task_type: str) -> Tuple[str, bool, bool]:
    """分析输出质量"""
    complexity = "simple"
    has_reasoning = False
    has_calculation = False
    
    # 检查推理词汇
    reasoning_keywords = ["因为", "所以", "因此", "由于", "推理", "逻辑", "前提", "结论",
                         "because", "therefore", "thus", "since", "reasoning", "logic",
                         "donc", "parce que", "reasoning", "logique",
                         "deshalb", "weil", "folgerung",
                         "поэтому", "потому что", "логика",
                         "だから", "なぜなら", "論理",
                         "그러므로", "왜냐하면", "추론",
                         "故", "是以", "因", "推"]
    
    if any(kw in response.lower() for kw in reasoning_keywords):
        has_reasoning = True
    
    # 检查计算
    if any(c in response for c in "0123456789+-*/=×÷"):
        has_calculation = True
    
    # 评估复杂度
    word_count = len(response.split())
    char_count = len(response)
    
    if task_type == "math":
        if has_calculation and has_reasoning and char_count > 200:
            complexity = "complex"
        elif has_calculation and char_count > 100:
            complexity = "medium"
    elif task_type == "logic":
        if has_reasoning and char_count > 200:
            complexity = "complex"
        elif has_reasoning and char_count > 100:
            complexity = "medium"
    elif task_type == "knowledge":
        if char_count > 300:
            complexity = "complex"
        elif char_count > 150:
            complexity = "medium"
    
    return complexity, has_reasoning, has_calculation


def run_language_test(model: str, language: str, prompt: str, task_type: str, run_number: int) -> TestResult:
    """运行单语言测试"""
    result = TestResult(
        model=model,
        language=language,
        language_name=MULTILINGUAL_PROMPTS[task_type].get(language, language),
        task_type=task_type,
        run_number=run_number,
        timestamp=datetime.now().isoformat(),
        input_length_chars=count_chars(prompt)
    )
    
    # 运行测试
    test_result = run_moxing_test(model, prompt, max_tokens=512)
    
    if test_result["success"]:
        result.success = True
        result.input_tokens = test_result.get("prompt_tokens", 0)
        result.output_tokens = test_result.get("completion_tokens", 0)
        result.total_tokens = test_result.get("total_tokens", 0)
        result.total_time_s = test_result.get("total_time", 0)
        result.tokens_per_second = test_result.get("tokens_per_second", 0)
        result.time_to_first_token_s = test_result.get("first_token_time", 0)
        result.output_length_chars = count_chars(test_result.get("response", ""))
        
        # 质量分析
        complexity, has_reasoning, has_calculation = analyze_output_quality(
            test_result.get("response", ""), task_type
        )
        result.output_complexity = complexity
        result.has_reasoning = has_reasoning
        result.has_calculation = has_calculation
    else:
        result.error = test_result.get("error", "Unknown error")
    
    return result


def calculate_statistics(results: List[TestResult]) -> List[LanguageStats]:
    """计算统计数据"""
    stats = []
    
    # 按语言 + 任务分组
    groups = {}
    for r in results:
        key = f"{r.language}_{r.task_type}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    for key, group_results in groups.items():
        language, task_type = key.split("_", 1)
        
        successful = [r for r in group_results if r.success]
        if not successful:
            continue
        
        stat = LanguageStats(
            language=language,
            language_name=successful[0].language_name,
            task_type=task_type,
            total_runs=len(group_results),
            successful_runs=len(successful),
            success_rate=len(successful) / len(group_results) * 100,
            
            avg_input_tokens=sum(r.input_tokens for r in successful) / len(successful),
            avg_output_tokens=sum(r.output_tokens for r in successful) / len(successful),
            avg_total_tokens=sum(r.total_tokens for r in successful) / len(successful),
            
            std_input_tokens=statistics.stdev([r.input_tokens for r in successful]) if len(successful) > 1 else 0,
            std_output_tokens=statistics.stdev([r.output_tokens for r in successful]) if len(successful) > 1 else 0,
            std_total_tokens=statistics.stdev([r.total_tokens for r in successful]) if len(successful) > 1 else 0,
            
            avg_speed=sum(r.tokens_per_second for r in successful) / len(successful),
            std_speed=statistics.stdev([r.tokens_per_second for r in successful]) if len(successful) > 1 else 0,
            avg_ttft=sum(r.time_to_first_token_s for r in successful) / len(successful),
            std_ttft=statistics.stdev([r.time_to_first_token_s for r in successful]) if len(successful) > 1 else 0,
        )
        
        stats.append(stat)
    
    return stats


def run_full_test(model: str, repetitions: int = 15) -> Dict:
    """运行完整的多语言测试"""
    print(f"\n{'='*80}")
    print(f"多语言 Token 消耗测试 - {model}")
    print(f"重复次数：{repetitions} | 语言：11 | 任务：3")
    print(f"总测试次数：{repetitions * 11 * 3}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for task_type in MULTILINGUAL_PROMPTS.keys():
        print(f"\n{'#'*80}")
        print(f"# 任务类型：{TEST_TASKS[task_type]['name']}")
        print(f"{'#'*80}\n")
        
        for lang, prompt in MULTILINGUAL_PROMPTS[task_type].items():
            lang_name = {"en":"English","zh":"中文","ja":"日本語","fr":"Français","ru":"Русский",
                        "de":"Deutsch","es":"Español","pt":"Português","it":"Italiano",
                        "ko":"한국어","lzh":"文言文"}.get(lang, lang)
            
            print(f"\n[{lang}] {lang_name} - {TEST_TASKS[task_type]['name']}")
            print(f"  Prompt({count_chars(prompt)} 字符): {prompt[:60]}...")
            
            lang_results = []
            for i in range(repetitions):
                print(f"    运行 {i+1:2d}/{repetitions}...", end=" ", flush=True)
                
                result = run_language_test(model, lang, prompt, task_type, i+1)
                lang_results.append(result)
                
                if result.success:
                    print(f"✅ {result.total_tokens} tokens | {result.tokens_per_second:5.1f} tok/s | TTFT: {result.time_to_first_token_s:.2f}s | {result.output_complexity}")
                else:
                    print(f"❌ {result.error}")
                
                time.sleep(0.5)  # 短暂等待
            
            all_results.extend(lang_results)
    
    # 生成统计
    stats = calculate_statistics(all_results)
    
    return {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "repetitions": repetitions,
        "languages": 11,
        "tasks": 3,
        "total_runs": len(all_results),
        "results": [asdict(r) for r in all_results],
        "statistics": [asdict(s) for s in stats]
    }


def save_results(data: Dict, filename: str):
    """保存测试结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存：{filename}")


def print_summary(all_test_data: List[Dict]):
    """打印对比摘要"""
    print("\n" + "="*80)
    print("测试结果对比摘要")
    print("="*80)
    
    for test_data in all_test_data:
        model = test_data["model"]
        stats = test_data["statistics"]
        
        print(f"\n{'='*80}")
        print(f"模型：{model}")
        print(f"{'='*80}")
        
        # 按任务类型分组
        for task_type in ["math", "logic", "knowledge"]:
            task_stats = [s for s in stats if s["task_type"] == task_type]
            if not task_stats:
                continue
            
            print(f"\n### {TEST_TASKS[task_type]['name']}\n")
            print(f"{'语言':<10} {'Avg In':>8} {'Avg Out':>8} {'Avg Tot':>8} {'Std':>6} {'Speed':>8} {'TTFT':>7} {'OK%':>5}")
            print("-"*80)
            
            # 按平均输出 tokens 排序
            sorted_stats = sorted(task_stats, key=lambda x: x["avg_output_tokens"])
            
            for s in sorted_stats:
                print(f"{s['language_name']:<10} {s['avg_input_tokens']:>8.1f} {s['avg_output_tokens']:>8.1f} {s['avg_total_tokens']:>8.1f} {s['std_total_tokens']:>6.1f} {s['avg_speed']:>8.1f} {s['avg_ttft']:>7.2f}s {s['success_rate']:>5.0f}%")


def main():
    """主函数"""
    print("="*80)
    print("多语言 Token 消耗对比测试 v2")
    print("="*80)
    print(f"\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n测试配置:")
    print(f"  语言：11 种 (中英日法俄德西班牙葡萄牙意大利韩 + 文言文)")
    print(f"  任务：3 类 (数学推理、逻辑推理、知识问答)")
    print(f"  重复：15 次/语言/任务")
    print(f"  总计：11 × 3 × 15 = 495 次测试/模型")
    print(f"\n测试模型:")
    print(f"  1. gemma3:1b")
    print(f"  2. carstenuhlig/omnicoder-9b:latest")
    
    models = ["gemma3:1b", "carstenuhlig/omnicoder-9b:latest"]
    all_test_data = []
    
    for model in models:
        start_time = time.time()
        test_data = run_full_test(model, repetitions=15)
        elapsed = time.time() - start_time
        
        test_data["elapsed_time_s"] = elapsed
        all_test_data.append(test_data)
        
        # 保存结果
        filename = f"multilingual_v2_{model.replace(':', '_').replace('/', '_')}.json"
        save_results(test_data, filename)
        
        print(f"\n模型 {model} 测试完成，耗时：{elapsed/60:.1f} 分钟")
    
    # 保存汇总
    save_results({"tests": all_test_data}, "multilingual_v2_summary.json")
    
    # 打印摘要
    print_summary(all_test_data)


if __name__ == "__main__":
    main()
