import torch
import io
import sys
# Força stdout usar UTF-8 em vez de charmap (Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch.nn.functional as F
import argparse
import time
import json
import re
import random
import string
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from src.model_v11.config import ModelConfig
from src.model_v11.model import ModelV2
from src.model_v11.inference import init_state, model_step, sample_token


class ToolExecutor:
    """Executes tool calls detected in model output. Simulates common tools."""

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate tool handler. Returns result dict."""
        tool_method = getattr(self, f"tool_{tool_name}", None)
        if tool_method is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return tool_method(**arguments)
        except TypeError as e:
            return {"error": f"Invalid arguments for {tool_name}: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def tool_calculate_tip(self, bill_amount: float, tip_percentage: float = 15.0) -> Dict:
        """Calculate tip and total bill."""
        tip = bill_amount * tip_percentage / 100
        return {
            "tip_amount": round(tip, 2),
            "total_bill": round(bill_amount + tip, 2),
            "tip_percentage": tip_percentage
        }

    def tool_convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """Convert between currencies using fixed rates."""
        rates = {
            "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5,
            "BRL": 5.05, "CAD": 1.36, "AUD": 1.55, "CHF": 0.88, "CNY": 7.24
        }
        if from_currency not in rates or to_currency not in rates:
            return {"error": f"Currency not supported: {from_currency} or {to_currency}"}
        converted = amount * (rates[to_currency] / rates[from_currency])
        return {
            "original": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "converted_amount": round(converted, 2)
        }

    def tool_calculate_bmi(self, weight: float, height: float) -> Dict:
        """Calculate BMI. Weight in kg, height in meters."""
        if height <= 0:
            return {"error": "Height must be positive"}
        bmi = weight / (height ** 2)
        category = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"
        return {
            "weight": weight,
            "height": height,
            "bmi": round(bmi, 1),
            "category": category
        }

    def tool_calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        """Calculate Euclidean distance between two points."""
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return {
            "point1": {"x": x1, "y": y1},
            "point2": {"x": x2, "y": y2},
            "distance": round(distance, 2)
        }

    def tool_calculate_age(self, birth_year: int, current_year: int = 2026) -> Dict:
        """Calculate age from birth year."""
        age = current_year - birth_year
        return {
            "birth_year": birth_year,
            "current_year": current_year,
            "age": age
        }

    def tool_calculate_discount(self, original_price: float, discount_percent: float) -> Dict:
        """Calculate discount and final price."""
        discount_amount = original_price * discount_percent / 100
        final_price = original_price - discount_amount
        return {
            "original_price": original_price,
            "discount_percent": discount_percent,
            "discount_amount": round(discount_amount, 2),
            "final_price": round(final_price, 2)
        }

    def tool_get_stock_price(self, symbol: str) -> Dict:
        """Simulate stock price lookup."""
        random.seed(hash(symbol) % 2**32)
        base_prices = {"AAPL": 150, "GOOGL": 140, "MSFT": 400, "TSLA": 180, "META": 320}
        base = base_prices.get(symbol.upper(), 100)
        price = base + random.uniform(-10, 10)
        return {
            "symbol": symbol.upper(),
            "price": round(price, 2),
            "currency": "USD",
            "timestamp": "2026-03-23"
        }

    def tool_generate_password(self, length: int = 12) -> Dict:
        """Generate a random password."""
        chars = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(random.choice(chars) for _ in range(max(8, length)))
        return {
            "password": password,
            "length": len(password),
            "contains_letters": True,
            "contains_numbers": True,
            "contains_special": True
        }

    def tool_generate_random_number(self, min_value: int = 0, max_value: int = 100) -> Dict:
        """Generate random number in range."""
        num = random.randint(min_value, max_value)
        return {
            "min": min_value,
            "max": max_value,
            "random_number": num
        }

    def tool_search_books(self, query: str, limit: int = 5) -> Dict:
        """Simulate book search."""
        books = [
            {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925},
            {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960},
            {"title": "1984", "author": "George Orwell", "year": 1949},
            {"title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813},
            {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "year": 1951},
        ]
        query_lower = query.lower()
        results = [b for b in books if query_lower in b["title"].lower() or query_lower in b["author"].lower()]
        return {
            "query": query,
            "count": len(results[:limit]),
            "books": results[:limit]
        }

    def tool_calculate_area(self, shape: str, **kwargs) -> Dict:
        """Calculate area of geometric shape."""
        shape = shape.lower()
        if shape == "rectangle":
            area = kwargs.get("width", 0) * kwargs.get("height", 0)
            return {"shape": "rectangle", "area": round(area, 2), "perimeter": round(2 * (kwargs.get("width", 0) + kwargs.get("height", 0)), 2)}
        elif shape == "circle":
            radius = kwargs.get("radius", 0)
            area = 3.14159 * (radius ** 2)
            return {"shape": "circle", "area": round(area, 2), "circumference": round(2 * 3.14159 * radius, 2)}
        elif shape == "triangle":
            base = kwargs.get("base", 0)
            height = kwargs.get("height", 0)
            area = 0.5 * base * height
            return {"shape": "triangle", "area": round(area, 2)}
        else:
            return {"error": f"Unknown shape: {shape}"}

    def tool_calculate_loan_payment(self, principal: float, annual_rate: float, years: int) -> Dict:
        """Calculate monthly loan payment."""
        monthly_rate = annual_rate / 100 / 12
        n_payments = years * 12
        if monthly_rate == 0:
            monthly_payment = principal / n_payments
        else:
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
        total_paid = monthly_payment * n_payments
        return {
            "principal": principal,
            "annual_rate": annual_rate,
            "years": years,
            "monthly_payment": round(monthly_payment, 2),
            "total_paid": round(total_paid, 2),
            "total_interest": round(total_paid - principal, 2)
        }

    def tool_send_email(self, to: str, subject: str, body: str) -> Dict:
        """Simulate email sending."""
        return {
            "status": "success",
            "to": to,
            "subject": subject,
            "message": "Email sent successfully",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def tool_create_todo(self, title: str, priority: str = "medium", due_date: str = None) -> Dict:
        """Simulate todo creation."""
        return {
            "status": "created",
            "id": f"todo_{random.randint(1000, 9999)}",
            "title": title,
            "priority": priority,
            "due_date": due_date,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def tool_create_note(self, title: str, content: str) -> Dict:
        """Simulate note creation."""
        return {
            "status": "created",
            "id": f"note_{random.randint(1000, 9999)}",
            "title": title,
            "content_length": len(content),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def tool_translate_text(self, text: str, target_language: str) -> Dict:
        """Simulate text translation (very basic)."""
        translations = {
            "spanish": {"hello": "hola", "goodbye": "adiós", "thank you": "gracias"},
            "french": {"hello": "bonjour", "goodbye": "au revoir", "thank you": "merci"},
            "german": {"hello": "hallo", "goodbye": "auf wiedersehen", "thank you": "danke"}
        }
        lang_dict = translations.get(target_language.lower(), {})
        words = text.lower().split()
        translated = " ".join(lang_dict.get(w, w) for w in words)
        return {
            "original": text,
            "target_language": target_language,
            "translated": translated
        }


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[ModelV2, ModelConfig, Optional[Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg_dict = checkpoint['config']
    cfg = ModelConfig(
        **{k: v for k, v in cfg_dict.items()
           if k in ModelConfig.__dataclass_fields__}
    )

    model = ModelV2(cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"[INFERENCE] Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"[INFERENCE] Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    tokenizer = None
    if cfg.vocab_size > 256:
        tp = Path("data/wikitext103/tokenizer.json")
        if tp.exists():
            from tokenizers import Tokenizer as HFTokenizer
            tokenizer = HFTokenizer.from_file(str(tp))
            chat_special = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
                            "<|tool_call|>", "<|tool_result|>", "<|thinking|>"]
            tokenizer.add_special_tokens(chat_special)
            print(f"[INFERENCE] BPE tokenizer loaded: vocab={tokenizer.get_vocab_size()}", flush=True)

    return model, cfg, tokenizer


def generate_text(
    model: ModelV2,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    device: torch.device = None,
    tokenizer: Optional[Any] = None
) -> str:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if tokenizer is not None:
        prompt_ids_list = tokenizer.encode(prompt).ids
    else:
        try:
            prompt_bytes = prompt.encode('utf-8', errors='replace')
        except Exception:
            prompt_bytes = prompt.encode('latin-1', errors='replace')
        prompt_ids_list = list(prompt_bytes)

    prompt_ids = torch.tensor(
        prompt_ids_list,
        dtype=torch.long,
        device=device
    )

    generated = list(prompt_ids_list)

    with torch.no_grad():
        state = init_state(model, batch_size=1, device=device)
        logits = None

        for token_id in prompt_ids:
            logits, state = model_step(
                model,
                token_id.view(1, 1),
                state
            )

        for step in range(max_tokens):
            next_token = sample_token(logits, temperature=temperature)
            generated.append(next_token)

            logits, state = model_step(
                model,
                torch.tensor([[next_token]], dtype=torch.long, device=device),
                state
            )

            if tokenizer is not None:
                current_text = tokenizer.decode(generated)
                end_markers = ['</code>', '</thinking>']
            else:
                current_text = bytes(generated).decode('utf-8', errors='replace')
                end_markers = [b'</code>', b'</thinking>']

            if len(generated) > len(prompt_ids_list) + 10:
                if tokenizer is not None:
                    if any(marker in current_text for marker in ['</code>', '</thinking>']):
                        break
                else:
                    if b'</code>' in bytes(generated[-10:]) or b'</thinking>' in bytes(generated[-10:]):
                        break

    if tokenizer is not None:
        output_text = tokenizer.decode(generated)
    else:
        try:
            output_text = bytes(generated).decode('utf-8', errors='replace')
        except Exception:
            output_text = bytes(generated).decode('latin-1', errors='replace')

    return output_text


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse <|tool_call|>{json}<|end|> from text. Returns (tool_name, arguments) or None."""
    pattern = r'<\|tool_call\|>\s*(\{.*?\})\s*<\|end\|>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        call_data = json.loads(match.group(1))
        tool_name = call_data.get("name", "").replace("-", "_")
        arguments = call_data.get("arguments", {})
        return (tool_name, arguments)
    except (json.JSONDecodeError, KeyError):
        return None


def generate_with_tools(
    model: ModelV2,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    device: torch.device = None,
    enable_tools: bool = False,
    debug: bool = False,
    tokenizer: Optional[Any] = None
) -> str:
    """Generate text with optional tool call detection and execution."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if tokenizer is not None:
        prompt_ids_list = tokenizer.encode(prompt).ids
    else:
        try:
            prompt_bytes = prompt.encode('utf-8', errors='replace')
        except Exception:
            prompt_bytes = prompt.encode('latin-1', errors='replace')
        prompt_ids_list = list(prompt_bytes)

    prompt_ids = torch.tensor(
        prompt_ids_list,
        dtype=torch.long,
        device=device
    )

    generated = list(prompt_ids_list)
    tool_executor = ToolExecutor() if enable_tools else None
    step_start = time.perf_counter() if debug else None
    vram_start = torch.cuda.memory_allocated() if (debug and torch.cuda.is_available()) else 0

    with torch.no_grad():
        state = init_state(model, batch_size=1, device=device)
        logits = None

        for token_id in prompt_ids:
            logits, state = model_step(model, token_id.view(1, 1), state)

        for step in range(max_tokens):
            next_token = sample_token(logits, temperature=temperature)
            generated.append(next_token)

            logits, state = model_step(
                model,
                torch.tensor([[next_token]], dtype=torch.long, device=device),
                state
            )

            if tokenizer is not None:
                current_text = tokenizer.decode(generated)
            else:
                try:
                    current_text = bytes(generated).decode('utf-8', errors='replace')
                except Exception:
                    current_text = bytes(generated).decode('latin-1', errors='replace')

            if enable_tools and '<|tool_call|>' in current_text:
                tool_result = parse_tool_call(current_text)
                if tool_result:
                    tool_name, args = tool_result
                    if debug:
                        print(f"[TOOL_CALL] {tool_name}({json.dumps(args)})", flush=True)
                    result = tool_executor.execute(tool_name, args)
                    if debug:
                        print(f"[TOOL_RESULT] {json.dumps(result)}", flush=True)
                    tool_result_str = f"<|tool_result|>{json.dumps(result)}<|end|>"
                    if tokenizer is not None:
                        tool_result_ids = tokenizer.encode(tool_result_str).ids
                        generated.extend(tool_result_ids)
                    else:
                        try:
                            tool_result_bytes = tool_result_str.encode('utf-8', errors='replace')
                        except Exception:
                            tool_result_bytes = tool_result_str.encode('latin-1', errors='replace')
                        generated.extend(list(tool_result_bytes))
                    if tokenizer is not None:
                        current_text = tokenizer.decode(generated)
                    else:
                        current_text = bytes(generated).decode('utf-8', errors='replace')

            if len(generated) > len(prompt_ids_list) + 10:
                if tokenizer is not None:
                    if '</code>' in current_text or '</thinking>' in current_text:
                        break
                else:
                    if b'</code>' in bytes(generated[-10:]) or b'</thinking>' in bytes(generated[-10:]):
                        break

    if debug and step_start:
        elapsed = time.perf_counter() - step_start
        n_generated = len(generated) - len(prompt_ids_list)
        tok_per_sec = n_generated / max(elapsed, 1e-6)
        vram_now = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        vram_delta_mb = (vram_now - vram_start) / (1024 ** 2) if torch.cuda.is_available() else 0
        print(f"[DEBUG] Generated {n_generated} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s, VRAM delta: {vram_delta_mb:.1f} MB", flush=True)

    if tokenizer is not None:
        output_text = tokenizer.decode(generated)
    else:
        try:
            output_text = bytes(generated).decode('utf-8', errors='replace')
        except Exception:
            output_text = bytes(generated).decode('latin-1', errors='replace')

    return output_text


def benchmark_vram(
    model: ModelV2,
    device: torch.device = None,
    seq_len: int = 256
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        x = torch.randint(0, 256, (1, seq_len), dtype=torch.long).to(device)
        _ = model(x)

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[BENCH] Peak VRAM: {peak_mb:.1f} MB", flush=True)
        return peak_mb
    else:
        print(f"[BENCH] VRAM benchmark not available on CPU", flush=True)
        return 0.0


def benchmark_throughput(
    model: ModelV2,
    device: torch.device = None,
    max_tokens: int = 200,
    tokenizer: Optional[Any] = None
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt = "def binary_search"

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    output = generate_text(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        device=device,
        tokenizer=tokenizer
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t_start

    if tokenizer is not None:
        prompt_ids = len(tokenizer.encode(prompt).ids)
        output_ids = len(tokenizer.encode(output).ids)
        n_generated = output_ids - prompt_ids
    else:
        prompt_bytes = prompt.encode('utf-8', errors='replace')
        output_bytes = output.encode('utf-8', errors='replace')
        n_generated = len(output_bytes) - len(prompt_bytes)

    tok_per_sec = n_generated / max(elapsed, 1e-6)

    print(f"[BENCH] Generated {n_generated} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s", flush=True)
    return tok_per_sec


def run_benchmark(
    model: ModelV2,
    device: torch.device,
    seq_len: int = 256,
    tokenizer: Optional[Any] = None
) -> Dict[str, float]:
    print(f"\n{'='*60}\nBENCHMARK\n{'='*60}", flush=True)

    vram_mb = benchmark_vram(model, device, seq_len)
    tok_per_sec = benchmark_throughput(model, device, max_tokens=200, tokenizer=tokenizer)

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"VRAM (inference): {vram_mb:.1f} MB")
    print(f"Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"{'='*60}\n", flush=True)

    return {
        "vram_mb": vram_mb,
        "tok_per_sec": tok_per_sec
    }


def interactive_mode(
    model: ModelV2,
    device: torch.device,
    enable_tools: bool = False,
    debug: bool = False,
    tokenizer: Optional[Any] = None
):
    print(f"\n{'='*60}\nINTERACTIVE MODE\n{'='*60}", flush=True)
    flags = []
    if enable_tools:
        flags.append("TOOLS")
    if debug:
        flags.append("DEBUG")
    flags_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"(Type 'quit' to exit, 'bench' to run benchmark, 'help' for info){flags_str}", flush=True)

    while True:
        try:
            prompt = input("\n[INPUT] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFERENCE] Exiting interactive mode", flush=True)
            break

        if prompt.lower() == 'quit':
            print("[INFERENCE] Exiting interactive mode", flush=True)
            break

        if prompt.lower() == 'help':
            print(f"\nINTERACTIVE MODE HELP:")
            print(f"  quit     - Exit interactive mode")
            print(f"  bench    - Run VRAM/throughput benchmark")
            print(f"  help     - Show this help message")
            print(f"  Flags enabled: {', '.join(flags) if flags else 'none'}")
            continue

        if prompt.lower() == 'bench':
            run_benchmark(model, device, seq_len=256, tokenizer=tokenizer)
            continue

        if not prompt:
            continue

        print(f"[OUTPUT] Generating...", flush=True)
        output = generate_with_tools(
            model,
            prompt,
            max_tokens=200,
            temperature=0.7,
            device=device,
            enable_tools=enable_tools,
            debug=debug,
            tokenizer=tokenizer
        )
        safe_output = output.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(f"[OUTPUT] {safe_output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Load trained model and generate text, benchmark, or interact with tool support."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to generate from"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run VRAM and throughput benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda, fallback to cpu if unavailable)"
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable tool use (detect tool_call, execute, inject result)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info (tok/s, VRAM, timings per generation)"
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", flush=True)
        return 1

    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[INFERENCE] CUDA not available, falling back to CPU", flush=True)
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    except RuntimeError:
        print(f"[ERROR] Failed to initialize device: {args.device}", flush=True)
        device = torch.device("cpu")

    try:
        model, cfg, tokenizer = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}", flush=True)
        return 1

    if args.benchmark:
        run_benchmark(model, device, seq_len=cfg.seq_len, tokenizer=tokenizer)
        if args.prompt:
            print(f"", flush=True)

    if args.prompt:
        try:
            output = generate_with_tools(
                model,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
                enable_tools=args.tools,
                debug=args.debug,
                tokenizer=tokenizer
            )
            safe_output = output.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            print(f"\n[OUTPUT]\n{safe_output}", flush=True)
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}", flush=True)
            return 1

    if not args.benchmark and not args.prompt:
        interactive_mode(model, device, enable_tools=args.tools, debug=args.debug, tokenizer=tokenizer)

    return 0


if __name__ == "__main__":
    exit(main())
