from litellm import completion, embedding
import os
import time
import csv
import json
from collections import deque
from typing import Dict, List
import math

# Set API Keys and Model Names
MODEL_COMPLETION = os.environ.get('LITELLM_COMPLETION', "openai/gpt-4o-mini")
MODEL_EMBEDDING = os.environ.get('LITELLM_EMBEDDING', "openai/text-embedding-ada-002")

# Set Variables
CSV_FILE = "task_data.csv"
OBJECTIVE = "Brainstorm and execute a social media brand targeting youth around AI with an ecommerce strategy."
YOUR_FIRST_TASK = "Develop a task list."

# Print OBJECTIVE
print("\033[96m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(OBJECTIVE)

# Ensure CSV file exists
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['result_id', 'task_name', 'result', 'embedding'])
        writer.writeheader()

# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text."""
    text = text.replace("\n", " ")
    try:
        response = embedding(
            model=MODEL_EMBEDDING,
            input=[text]
        )
        return response.data[0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def context_agent(query: str, n: int) -> List[str]:
    """Retrieve top N most similar tasks based on the query embedding."""
    query_embedding = get_embedding(query)
    results = []

    try:
        with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                stored_embedding = json.loads(row['embedding'])
                similarity = cosine_similarity(query_embedding, stored_embedding)
                results.append({
                    'task': row['task_name'],
                    'result': row['result'],
                    'similarity': similarity
                })
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    # Sort by similarity descending and return top n
    sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    top_results = sorted_results[:n]
    return [item['task'] for item in top_results]

def task_creation_agent(objective: str, result: Dict, task_description: str, incomplete_tasks: List[str]) -> List[Dict]:
    """Create new tasks based on the execution result."""
    prompt = (
        f"You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}. "
        f"The last completed task has the result: {result}. This result was based on this task description: {task_description}. "
        f"These are incomplete tasks: {', '.join(incomplete_tasks)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. "
        f"Return the tasks as a JSON array."
    )
    try:
        response = completion(
            model=MODEL_COMPLETION,
            messages=[
                {"role": "system", "content": "You are a helpful task creation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        new_tasks_text = response.choices[0].message.content.strip()
        new_tasks = json.loads(new_tasks_text)
        if isinstance(new_tasks, list):
            return [{"task_name": task.strip()} for task in new_tasks]
    except json.JSONDecodeError:
        # Fallback: split by lines if JSON parsing fails
        new_tasks = new_tasks_text.split('\n')
        return [{"task_name": task.strip()} for task in new_tasks if task.strip()]
    except Exception as e:
        print(f"Error in task_creation_agent: {e}")
    return []

def prioritization_agent(this_task_id: int):
    """Reprioritize the task list based on the objective."""
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = this_task_id + 1
    prompt = (
        f"""You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. 
        Consider the ultimate objective of your team: {OBJECTIVE}. Do not remove any tasks. 
        Return the result as a numbered list, like:
        1. First task
        2. Second task
        Start the task list with number {next_task_id}."""
    )
    try:
        response = completion(
            model=MODEL_COMPLETION,
            messages=[
                {"role": "system", "content": "You are a helpful task prioritization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        new_tasks_text = response.choices[0].message.content.strip()
        new_tasks = new_tasks_text.split('\n')
        task_list = deque()
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                task_list.append({"task_id": task_id, "task_name": task_name})
    except Exception as e:
        print(f"Error in prioritization_agent: {e}")

def execution_agent(objective: str, task: str) -> str:
    """Execute the given task based on the objective and context."""
    context = context_agent(query=objective, n=5)
    prompt = f"Objective: {objective}\nTask: {task}\nResponse:"
    try:
        response = completion(
            model=MODEL_COMPLETION,
            messages=[
                {"role": "system", "content": "You are an AI who performs one task based on the given objective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in execution_agent: {e}")
        return ""

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)

# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(f"{t['task_id']}: {t['task_name']}")

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(f"{task['task_id']}: {task['task_name']}")

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in CSV
        enriched_result = {'data': result}  # Enrichment can be added here if needed
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']  # Extract the actual result from the dictionary
        result_embedding = get_embedding(vector)

        # Append to CSV
        try:
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['result_id', 'task_name', 'result', 'embedding'])
                writer.writerow({
                    'result_id': result_id,
                    'task_name': task['task_name'],
                    'result': result,
                    'embedding': json.dumps(result_embedding)
                })
        except Exception as e:
            print(f"Error writing to CSV: {e}")

        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            objective=OBJECTIVE,
            result=enriched_result['data'],
            task_description=task["task_name"],
            incomplete_tasks=[t["task_name"] for t in task_list]
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)

        prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again
