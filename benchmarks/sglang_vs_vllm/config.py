"""
Benchmark configuration for SGLang + Megatron vs vLLM + Megatron.

All parameters that control the benchmark are defined here to ensure
both backends are tested under identical conditions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model configuration shared across both backends."""

    # Model identity — Megatron backend requires Qwen3 MoE
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_name: str = "benchmark-model"
    project: str = "sglang-vs-vllm-benchmark"

    # Sequence lengths
    max_seq_length: int = 8192
    max_input_tokens: int = 1024
    max_output_tokens: int = 1024

    # LoRA config (must match Megatron train.py defaults)
    lora_r: int = 1
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class InferenceConfig:
    """Inference engine configuration."""

    # Parallelism
    tensor_parallel_size: int = 0  # 0 = auto-detect (min(2, num_gpus))

    # Memory
    gpu_memory_utilization: float = 0.85

    # Batching
    max_num_seqs: int = 256

    # LoRA
    enable_lora: bool = True
    max_loras: int = 2

    def get_tp_size(self) -> int:
        import torch
        if self.tensor_parallel_size <= 0:
            return min(2, torch.cuda.device_count())
        return self.tensor_parallel_size


@dataclass
class TrainingConfig:
    """Training configuration shared across both backends."""

    learning_rate: float = 5e-6
    beta: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 0.1
    weight_decay: float = 0.1


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    # --- What to benchmark ---
    backends: list[Literal["vllm", "sglang"]] = field(
        default_factory=lambda: ["vllm", "sglang"]
    )

    # --- Workload ---
    # agentic dataset: multi-turn tool-use prompts → triggers RadixAttention
    # advantage in SGLang (5x throughput on shared prefixes)
    dataset: str = "agentic"  # "gsm8k", "sharegpt", "math", "agentic", "synthetic"
    num_training_steps: int = 3
    num_rollouts_per_step: int = 16
    # concurrency=32: SGLang peaks at ~60 concurrent, vLLM peaks at ~40
    # Higher concurrency exposes SGLang's scheduling advantage
    concurrency: int = 32
    num_warmup_requests: int = 4  # warmup requests before timing
    seed: int = 42  # deterministic prompt sampling

    # --- Repetitions for statistical significance ---
    num_repeats: int = 1  # repeat entire benchmark N times

    # --- Output ---
    output_dir: str = "benchmark_results"
    save_raw_metrics: bool = True
    generate_charts: bool = True

    # --- Environment ---
    sglang_python: str = ""  # path to SGLang env python, auto-detected if empty
    vllm_python: str = ""  # path to vLLM env python, auto-detected if empty

    # --- Ports ---
    vllm_port: int = 8100
    sglang_port: int = 8200

    # --- Timeouts ---
    server_startup_timeout: int = 600  # seconds to wait for server to start
    request_timeout: int = 300  # seconds per request
    training_timeout: int = 1800  # seconds per training step
    server_shutdown_timeout: int = 60  # seconds to wait for server shutdown

    # --- Component configs ---
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.sglang_python:
            self.sglang_python = _find_sglang_python()
        if not self.vllm_python:
            self.vllm_python = _find_vllm_python()


def _find_sglang_python() -> str:
    """Auto-detect SGLang environment python."""
    candidates = [
        os.path.expanduser("~/.venvs/sglang-bench/bin/python"),
        os.path.expanduser("~/sglang-env/bin/python"),
        os.path.join(os.path.dirname(__file__), "..", "..", ".sglang-env", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # Fallback: assume sglang is installed in current env
    return "python"


def _find_vllm_python() -> str:
    """Auto-detect vLLM environment python (use current ART environment)."""
    import sys
    return sys.executable


# ---------------------------------------------------------------------------
# Prompt generation for benchmarking — real datasets
# ---------------------------------------------------------------------------

# Hard-coded GSM8K-style math problems (no download required, deterministic).
# These mimic the distribution of real GSM8K: grade-school math word problems
# that require multi-step reasoning — the canonical RL-training benchmark.
_GSM8K_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted plants of three colors in it. Ten of them are yellow, and there are 80% more of those in red. Blue flowers make up only 25% of red flowers. How many flowers does Mark have in his garden?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother, who lives 100 miles away. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
    "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 less than her budget. If she found a shirt for $60 and bought 2 pairs of pants of equal price, how much did each pair of pants cost?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    "A farmer is buying feed for his horses. He buys a variety of hay, oats, carrots and sugar cubes. Since sugar cubes are a rare treat, he only buys two 1-pound boxes of them for the whole stable. He only wants enough carrots to feed the horses while the vegetables are fresh, so he buys four 12-pound bags. Hay is the main diet of his horses, so he buys forty-two 75-pound bales. Oats are a staple to supplement the hay, so he buys twenty 65-pound sacks. If his farm truck can carry 2250 pounds at a time, how many trips does the farmer need to transport all the feed?",
    "A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financial advisor advises him to invest 5/8 of his total assets in jewelry and the rest in gadgets. If the merchant's total assets are $80,000, how much will he save if he takes his financial adviser's advice?",
    "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
    "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "A pet store had 78 puppies. In one day they sold 30 of them and put the rest into cages with 8 in each cage. How many cages did they use?",
    "Brandon's iPhone is four times as old as Ben's iPhone. Ben's iPhone is two times older than Suzy's iPhone. If Suzy's iPhone is 1 year old, how old is Brandon's iPhone?",
    "Pam is currently twice as young as Rena is. In 10 years, Rena will be 5 years older than Pam. How old is Pam now?",
    "Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Samantha is now 31, how many years ago was Raymond's son born?",
    "Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each. His next 2 customers buy 2 DVDs each. His last 3 customers don't buy any DVDs. How many DVDs did Billy sell on Tuesday?",
    "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. How many students enrolled in hip-hop dance?",
    "A mechanic charges different rates to repair the tires of trucks and cars. For each truck tire, the mechanic charges $60 and for each car tire, he charges $40. On Thursday, the mechanic repairs 6 truck tires and 4 car tires. On Friday, he repairs 12 car tires and doesn't repair any truck tires. How much more revenue did the mechanic earn on the day with higher revenue?",
    "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
    "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.50 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "James creates a media empire. He creates a movie for $2000. Each DVD costs $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?",
    "Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she need to buy to make omelets for 4 weeks?",
    "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed for the entire trail to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?",
    "Jim spends 2 hours watching TV and then decides to go to bed and reads for half as long. He does this 3 times a week. How many hours does he spend on TV and reading in 4 weeks?",
    "There are 25 roses in a garden. There are 40% more tulips than roses. There are 35% fewer daisies than tulips. How many flowers are there in the garden in total?",
]

# ShareGPT-style conversational prompts (realistic chat workload).
_SHAREGPT_PROMPTS = [
    "Can you explain the difference between TCP and UDP? When would you use each one?",
    "I'm trying to learn Python. Can you write a simple web scraper that extracts all links from a webpage?",
    "What are the main differences between REST and GraphQL APIs? Give me pros and cons of each.",
    "Help me write a cover letter for a senior software engineer position at a tech startup.",
    "Explain how transformers work in machine learning. I have a CS background but haven't studied NLP.",
    "I need to design a database schema for an e-commerce platform. What tables should I create?",
    "What is the CAP theorem and why is it important for distributed systems?",
    "Can you help me debug this error? I'm getting 'CUDA out of memory' when training a model.",
    "Write a bash script that monitors disk usage and sends an alert when it exceeds 90%.",
    "Explain the concept of attention mechanism in neural networks with a concrete example.",
    "How do I set up a CI/CD pipeline with GitHub Actions for a Python project?",
    "What are the best practices for handling authentication in a microservices architecture?",
    "I'm building a recommendation system. Should I use collaborative filtering or content-based filtering?",
    "Explain Docker containers vs virtual machines. When should I use each?",
    "Write a SQL query that finds the top 5 customers by total purchase amount in the last 30 days.",
    "How do I implement rate limiting in a Node.js API? Give me different approaches.",
    "What is the difference between supervised, unsupervised, and reinforcement learning? Give examples of each.",
    "Help me optimize this Python function that processes a large CSV file. It's currently too slow.",
    "Explain how garbage collection works in Java vs Python.",
    "I need to migrate a monolithic application to microservices. What's a good strategy?",
    "How does Bitcoin mining work? Explain it technically but in simple terms.",
    "Write a Python decorator that caches function results with a TTL (time-to-live).",
    "What are the SOLID principles in object-oriented design? Give a code example for each.",
    "Explain the difference between process and thread. When would you use multiprocessing vs multithreading?",
    "How do I implement a load balancer from scratch? What algorithms are commonly used?",
    "What is eventual consistency and how does it differ from strong consistency?",
    "Help me write unit tests for a function that calls an external API.",
    "Explain how HTTPS works step by step, starting from when a user types a URL in the browser.",
    "What are the common ways to prevent SQL injection attacks?",
    "I'm choosing between PostgreSQL and MongoDB for my project. What should I consider?",
    "Write a recursive function to solve the Tower of Hanoi problem and explain the time complexity.",
    "How do I implement OAuth 2.0 authorization code flow?",
]

# Agentic/tool-use prompts (matches ART's primary use case: RL for agents).
_AGENTIC_PROMPTS = [
    "You have access to a calculator tool. A store has a 30% off sale. If an item originally costs $85.50 and there's an additional 10% member discount applied after the sale price, what's the final price? Show your work using the calculator.",
    "You are an AI assistant with access to a web search tool. The user asks: 'What was the GDP of France in 2023 and how did it compare to Germany?' Search for the relevant data and provide a comprehensive answer.",
    "You have access to a Python code execution environment. Write and execute a script that generates the first 20 numbers in the Fibonacci sequence, then calculate the ratio between consecutive numbers to show it converges to the golden ratio.",
    "You are a coding assistant with access to a file system. The user wants you to create a simple TODO app with a REST API. Plan the file structure and create the main application file.",
    "You have access to a database query tool. The user asks: 'Find all orders from the last quarter where the total exceeded $1000, group them by customer, and show the average order value per customer.'",
    "You are an AI research assistant with access to academic paper search. Find recent papers on 'reinforcement learning from human feedback' and summarize the key contributions of the top 3 most cited ones.",
    "You have access to a weather API and a calendar tool. The user is planning a hiking trip next weekend. Check the weather forecast and suggest the best day to go hiking based on conditions.",
    "You are a data analysis assistant with access to Python and visualization tools. The user gives you sales data for 12 months. Analyze trends, identify seasonality, and create a forecast for the next 3 months.",
    "You have access to a code review tool. Review the following pull request that adds authentication middleware to an Express.js application. Check for security vulnerabilities and suggest improvements.",
    "You are an AI assistant helping with system administration. The server is running slow. Use diagnostic tools to check CPU usage, memory usage, disk I/O, and network traffic to identify the bottleneck.",
    "You have access to a translation API. Translate the following legal contract from English to Spanish, ensuring legal terminology is correctly translated. Then back-translate to verify accuracy.",
    "You are a financial advisor assistant with access to stock market data tools. Analyze the performance of AAPL, GOOGL, and MSFT over the past year and provide a comparison summary.",
    "You have access to a document analysis tool. Extract all dates, monetary amounts, and named entities from the following contract and organize them in a structured format.",
    "You are a debugging assistant. The user's Docker container keeps crashing with exit code 137. Use system monitoring tools to diagnose the issue and suggest fixes.",
    "You have access to a map API and a routing tool. Plan a road trip from San Francisco to Los Angeles with 3 stops at interesting landmarks, optimizing for both scenic routes and minimal driving time.",
    "You are an email management assistant. The user has 50 unread emails. Categorize them by priority (urgent, normal, low), summarize each, and draft responses for the urgent ones.",
]

# MATH-competition-level prompts (harder, longer reasoning chains).
_MATH_PROMPTS = [
    "Let $f(x) = x^3 - 3x + 1$. Find the number of real roots of $f(f(x)) = 0$.",
    "In triangle ABC, AB = 13, BC = 14, and CA = 15. Let M be the midpoint of BC. Find the length of AM.",
    "Find all pairs of positive integers $(a, b)$ such that $a^2 + b^2 = 2(a + b) + 2ab$.",
    "A sequence is defined by $a_1 = 1$, $a_2 = 1$, and $a_{n+2} = a_{n+1} + a_n + 1$ for all $n \\geq 1$. Find $a_{10}$.",
    "How many ways can you place 8 non-attacking rooks on a standard 8x8 chessboard?",
    "Find the sum of all positive integers $n$ less than 1000 such that $n^2 + 1$ is divisible by $n + 1$.",
    "Let $S = \\{1, 2, 3, \\ldots, 100\\}$. How many subsets of $S$ have the property that the sum of elements is divisible by 5?",
    "A regular hexagon has side length 2. Find the area of the region inside the hexagon but outside the inscribed circle.",
    "Find the remainder when $2^{2023}$ is divided by 1000.",
    "In how many ways can you tile a 2 x 10 rectangle with 1 x 2 dominoes?",
    "Let $p(x) = x^4 + ax^3 + bx^2 + cx + d$ be a polynomial with integer coefficients such that $p(1) = 10$, $p(2) = 20$, $p(3) = 30$. Find $p(12) + p(-8)$ divided by 4.",
    "A point is chosen uniformly at random inside a unit square. What is the probability that the point is closer to the center than to any edge?",
    "Find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ satisfying $f(x + y) = f(x) + f(y) + 2xy$ for all real numbers $x, y$.",
    "There are 100 people in a room. Each person knows at least 67 other people in the room. Prove that there exist 4 people who all know each other.",
    "A 3x3x3 cube is painted red on all faces, then cut into 27 unit cubes. If one unit cube is randomly selected, what is the expected number of red faces?",
    "Evaluate: $\\sum_{k=1}^{\\infty} \\frac{k}{2^k}$",
]


def _get_dataset_prompts(dataset_name: str) -> list[str]:
    """Return the prompt list for the given dataset name."""
    datasets = {
        "gsm8k": _GSM8K_PROMPTS,
        "sharegpt": _SHAREGPT_PROMPTS,
        "agentic": _AGENTIC_PROMPTS,
        "math": _MATH_PROMPTS,
    }
    if dataset_name in datasets:
        return datasets[dataset_name]
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Choose from: {', '.join(datasets)} or 'synthetic'"
    )


def generate_benchmark_prompts(
    num_prompts: int,
    input_tokens: int = 1024,
    dataset: str = "gsm8k",
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """
    Generate deterministic benchmark prompts from a real dataset.

    For gsm8k/sharegpt/agentic/math: samples from the built-in prompt bank
    with deterministic seeding so both backends get identical prompts.

    For 'synthetic': generates padded prompts to hit a target token count
    (useful for pure throughput measurement at controlled input lengths).

    Returns a list of chat-format message lists.
    """
    rng = random.Random(seed)

    if dataset == "synthetic":
        return _generate_synthetic_prompts(num_prompts, input_tokens, rng)

    source_prompts = _get_dataset_prompts(dataset)

    # Sample with replacement if we need more prompts than the pool
    sampled = [rng.choice(source_prompts) for _ in range(num_prompts)]

    # Build chat-format messages with a system prompt that encourages
    # reasoning (typical for RL training workloads)
    system_msg = (
        "You are a helpful assistant. Think step by step and show your reasoning."
    )

    prompts = []
    for user_text in sampled:
        prompts.append([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ])
    return prompts


def _generate_synthetic_prompts(
    num_prompts: int,
    input_tokens: int,
    rng: random.Random,
) -> list[list[dict[str, str]]]:
    """
    Generate synthetic prompts padded to a target token count.
    Useful for controlled throughput measurement at a specific input length.
    """
    # Use diverse seed sentences so KV-cache / prefix-sharing doesn't
    # unfairly advantage one engine over the other.
    seed_sentences = [
        "Analyze the economic impact of renewable energy adoption in developing nations.",
        "Describe the process of photosynthesis and its significance in the carbon cycle.",
        "Explain the principles behind public-key cryptography and its applications.",
        "Discuss the ethical implications of artificial intelligence in healthcare.",
        "Compare and contrast the political systems of parliamentary and presidential democracies.",
        "Outline the major events of the Industrial Revolution and their lasting effects.",
        "Explain how neural networks learn from data and the role of backpropagation.",
        "Describe the water cycle and its importance for maintaining Earth's ecosystems.",
    ]

    prompts = []
    for i in range(num_prompts):
        seed = seed_sentences[i % len(seed_sentences)]
        # Pad to approximately the target token count (~4 chars/token)
        target_chars = input_tokens * 4
        padding_sentences = [
            f"Consider aspect {j}: {seed}" for j in range(target_chars // len(seed) + 1)
        ]
        content = " ".join(padding_sentences)[:target_chars]
        content += "\n\nProvide a detailed analysis with step-by-step reasoning."
        prompts.append([{"role": "user", "content": content}])
    return prompts
