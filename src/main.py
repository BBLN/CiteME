import json
import pandas as pd
from rich.console import Console
from retriever.agent import (
    LLMSelfAskAgentPydantic,
    LLMNoSearch,
    OutputSearchOnly,
    Output,
)
from utils.str_matcher import find_match_psr, find_multi_match_psr
from utils.tokens import num_tokens_from_string
from datetime import datetime
from time import time
from rich.progress import track
from retriever.llm_base import DEFAULT_TEMPERATURE
import traceback
import argparse
import wandb
import os

parser = argparse.ArgumentParser(description="Run few-shot search on CiteMe dataset")
parser.add_argument("--prompt_name", type=str, default="one_shot_search", help="Prompt template")
parser.add_argument("--result_file", type=str, default="few-shot-search-4o.json")
parser.add_argument("--max_actions", type=int, default=15)
parser.add_argument("--search_provider", type=str, default="SemanticScholarSearchProvider", choices=[
    "SemanticScholarSearchProvider", "SemanticScholarWebSearchProvider", "RAGProvider"])
parser.add_argument("--model", type=str, default="phi")
parser.add_argument("--peft_adapter", type=str)
parser.add_argument("--top_p", type=float)
parser.add_argument("--executor", type=str, default="LLMSelfAskAgentPydantic")
args = parser.parse_args()

generation_kwargs = {}
if args.top_p:
    generation_kwargs.update(dict(do_sample=True, top_p=args.top_p))

slurm_job_id = os.environ.get("SLURM_JOB_ID")
run = wandb.init(project="CiteME", config=args, group=f"{slurm_job_id}")

# -- Modify the following variables as needed --
TITLE_SEPERATOR = "[TITLE_SEPARATOR]"
RESULT_FILE_NAME = args.result_file
INCREMENTAL_SAVE = True

metadata = {
    "model": args.model,
    "peft_adapter": args.peft_adapter,
    "temperature": DEFAULT_TEMPERATURE,
    "executor": args.executor,
    "search_provider": args.search_provider,
    "prompt_name": args.prompt_name,  # See prompt names in retriever/prompt_templates
    "actions": "search_relevance,search_citation_count,select",
    "search_limit": 10,
    "threshold": 0.8,
    "execution_date": datetime.now().isoformat(),
    "only_open_access": False,
    "dataset_split": "all",
    "max_actions": args.max_actions,
}
# -- NO USER EDITABLE CODE BELOW THIS LINE --

console = Console()

## Load the dataset
c = pd.read_csv("hf://datasets/bethgelab/CiteME/CiteME.tsv", sep="\t")
c.set_index("id", inplace=True)
if metadata["dataset_split"] == "all":
    pass
elif metadata["dataset_split"] == "test":
    c = c[c["split"] == 'test']
elif metadata["dataset_split"] == "train":
    c = c[c["DevSet"] == 'train']
else:
    raise ValueError("Invalid dataset split")

## Select executor
if metadata["executor"] == "LLMSelfAskAgentPydantic":
    # Select action space
    if metadata["actions"] == "search_relevance,search_citation_count,read,select":
        pdo = Output
    elif metadata["actions"] == "search_relevance,search_citation_count,select":
        pdo = OutputSearchOnly
    else:
        raise ValueError("Invalid actions")
    agent = LLMSelfAskAgentPydantic(
        only_open_access=metadata["only_open_access"],
        search_limit=metadata["search_limit"],
        model_name=metadata["model"],
        peft_adapter=metadata["peft_adapter"],
        temperature=metadata["temperature"],
        prompt_name=metadata["prompt_name"],
        pydantic_object=pdo,
        generation_kwargs=generation_kwargs,
    )
elif metadata["executor"] == "LLMNoSearch":
    metadata["actions"] = None
    metadata["search_provider"] = None
    metadata["search_limit"] = 0
    metadata["use_web_search"] = False
    metadata["max_actions"] = 0
    assert metadata["prompt_name"].endswith("no_search"), "Only no search prompt are supported when executor is LLMNoSearch"
    agent = LLMNoSearch(
        model_name=metadata["model"],
        temperature=metadata["temperature"],
        prompt_name=metadata["prompt_name"],
    )
else:
    raise ValueError("Invalid executor")

console.log(f"AGENT BACKBONE: [bold green]{metadata['model']}")
console.log(f"PROMPT:         [bold green]{metadata['prompt_name']}")
console.log(f"ACTIONS:        [bold blue]{metadata['actions']}")

results = []
total_citations = 0
total_correct = 0
total_in_search = 0
eval_table = wandb.Table(columns=["split","correct","in_search"])
for cid, citation in track(
    c.iterrows(), description="Processing citations", total=len(c), console=console
):
    agent.reset([citation["source_paper_title"]], max_actions=metadata["max_actions"])
    
    start_time = time()
    citation_text = citation["excerpt"]
    target_titles = citation["target_paper_title"].split(TITLE_SEPERATOR)
    year = int(citation["year"] if not pd.isna(citation["year"]) else 2024)
    result_data = {
        "id": cid,
        "cited_paper_titles": target_titles,
        "excerpt": citation_text,
        "split": citation["split"],
    }
    try:
        selection = agent(citation_text, f"{year}")
        result_data["selected"] = selection.model_dump()
        is_in_search = find_multi_match_psr(
            sum(agent.paper_buffer, []), target_titles, threshold=metadata["threshold"]
        )
        result_data["is_in_search"] = is_in_search[1] if is_in_search else None
        correct = (
            find_match_psr(target_titles, selection.title, metadata["threshold"])
            is not None
        )
        result_data["is_correct"] = correct
        result_data["status"] = "success"
        result_data["error"] = None

        console.log(
            f"{cid:3d}: {target_titles[0]}\n"
            + f"     [{'green' if correct else 'red'}]{selection.title}\n"
            + f"     [black]search: {'✅' if is_in_search is not None else '❌'}, selection: {'✅' if correct else '❌'}"
        )
    except Exception as e:
        result_data["selected"] = None
        result_data["is_in_search"] = None
        result_data["is_correct"] = False
        result_data["status"] = "error"
        result_data["error"] = str(e)
        console.log(f"[red]{cid:3d}: {target_titles[0]}\n     [bold]{e}")
        print(traceback.format_exc())
        print(agent.get_history())

    result_data["duration"] = time() - start_time
    result_data["papers"] = agent.get_paper_buffer()
    result_data["history"] = agent.get_history(ignore_system_messages=False)
    if metadata["model"].startswith("gpt"):
        result_data["tokens"] = {
            "input": sum(
                [
                    num_tokens_from_string(m["content"], metadata["model"])
                    for m in result_data["history"]
                    if m["role"] != "assistant"
                ]
            ),
            "output": sum(
                [
                    num_tokens_from_string(m["content"], metadata["model"])
                    for m in result_data["history"]
                    if m["role"] == "assistant"
                ]
            ),
        }

    results.append(result_data)
    eval_table.add_data(result_data["split"], result_data["is_correct"], result_data["is_in_search"])
    if result_data["is_correct"]:
        total_correct += 1
    if result_data["is_in_search"]:
        total_in_search += 1
    total_citations += 1
    run.log({
             'total_citations': total_citations, 
             'total_correct': total_correct, 
             'total_in_search': total_in_search, 
             'citations': 1, 'correct': int(result_data["is_correct"]), 'in_search': 1 if result_data["is_in_search"] else 0,
             'split': result_data["split"],
             'accuracy': float(total_correct/total_citations)})
    if INCREMENTAL_SAVE:
        with open(RESULT_FILE_NAME, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=4)

# summary metrisc
run.summary["correct"] = sum(r["is_correct"] for r in results)
run.summary["total"] = len(results)
run.summary["correct_search"] = sum(not not r["is_in_search"] for r in results)

with open(RESULT_FILE_NAME, "w") as f:
    json.dump({"metadata": metadata, "results": results}, f, indent=4)

run.log_artifact(RESULT_FILE_NAME)

metrics_table = wandb.Table(columns=["split", "correct", "total", "correct_search"])
for split in ["all", "train", "test"]:
    metrics_table.add_data(split,
        sum(r["is_correct"] for r in results if r["split"] == split),
        len([r for r in results if r["split"] == split]),
        sum(r["is_in_search"] for r in results if r["split"] == split and r["is_in_search"]))
run.log({'metrics': metrics_table, 'eval': eval_table})
