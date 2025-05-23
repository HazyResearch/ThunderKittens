import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev

import psutil
import pydra
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    prompt_len: int | None = 64
    prompt: str | None = None
    output_len: int = 128
    batch_size: int = 1
    num_warmup: int = 5
    num_iters: int = 10
    env: str | None = None
    conda_activate_path: Path = Path("~/jj/miniconda3/bin/activate").expanduser()
    port: int = 10210
    launch: str | None = None
    api_key: str = "letmein"
    temperature: float = 0.0

    def finalize(self):
        if self.prompt_len is None:
            assert self.prompt is not None
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.prompt_len = len(tokenizer.encode(self.prompt))

    def l1(self):
        self.model = "meta-llama/Llama-3.2-1B-Instruct"

    def l8(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"


def prepend_conda_activate(command: str, activate_path: str, env: str):
    return f"source {activate_path} && conda activate {env} && {command}"


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            child.kill()

        parent.kill()

    except psutil.NoSuchProcess:
        pass


def wait_for_startup(
    process: subprocess.Popen,
    port: int,
    model: str,
    max_retries: int = 500,
    retry_seconds: float = 2,
):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="letmein",
        max_retries=0,
        timeout=20,
    )

    for i in range(max_retries):
        if process.poll() is not None:
            raise RuntimeError(f"Server crashed with returncode {process.returncode}")

        try:
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "tell me a funny joke about cookies"}
                ],
                max_tokens=10,
            )
            return
        except Exception:
            print(f"Server not yet started (attempt {i}) retrying...")
            time.sleep(retry_seconds)

    raise RuntimeError(f"Server not started after {max_retries} attempts.")


@contextmanager
def launch_server(config: ScriptConfig):
    if config.launch is None:
        yield None
        return

    if config.launch == "input":
        config.launch = input("Enter the command to launch the server: ")

    command = config.launch
    if config.env is not None:
        command = prepend_conda_activate(
            command, config.conda_activate_path, config.env
        )

    print(f"Starting server with command: '{command}'")
    server_process = subprocess.Popen(command, shell=True, executable="/bin/bash")
    print(f"Started server with pid {server_process.pid}")

    try:
        wait_for_startup(
            server_process, config.port, config.model, max_retries=500, retry_seconds=2
        )
        yield
    finally:
        print(f"Killing server (pid {server_process.pid})...")
        kill_process_tree(server_process.pid)
        print("Done killing server.")


def go(config: ScriptConfig, client: OpenAI, n_in: int, n_out: int, batch_size: int):
    times = []

    for i in tqdm(range(config.num_warmup + config.num_iters)):
        start = time.time()
        resp = client.completions.create(
            model=config.model,
            prompt=[0] * n_in,
            max_tokens=n_out,
            temperature=config.temperature,
            n=batch_size,
            extra_body={"ignore_eos": True},
        )
        end = time.time()
        assert resp.usage.completion_tokens == batch_size * n_out

        if i >= config.num_warmup:
            times.append(end - start)

    return mean(times), stdev(times)


def main(config: ScriptConfig):
    print(f"Running with config: {config.to_dict()}")

    with launch_server(config):
        client = OpenAI(
            api_key="fake-key",
            base_url=f"http://0.0.0.0:{config.port}/v1",
        )

        baseline_mean, baseline_stdev = go(
            config,
            client,
            n_in=config.prompt_len,
            n_out=1,
            batch_size=config.batch_size,
        )

        run_mean, run_stdev = go(
            config,
            client,
            n_in=config.prompt_len,
            n_out=config.output_len,
            batch_size=config.batch_size,
        )

        print(f"Baseline: {baseline_mean} ± {baseline_stdev}")
        print(f"Run: {run_mean} ± {run_stdev}")

        diff = run_mean - baseline_mean
        tokens = config.batch_size * (config.output_len - 1)
        tps = tokens / diff
        print(f"Throughput: {tps} tokens/s")


if __name__ == "__main__":
    pydra.run(main)
