import json
import random
import re
import sqlite3
import os
import anthropic

def load_json_file(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode the file {file_path} with the attempted encodings: {encodings}")

def scramble_sequence(data):
    keys = list(data.keys())
    random.shuffle(keys)
    return {key: data[key] for key in keys}

def init_claude():
    api_key = "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"
    return anthropic.Anthropic(api_key=api_key)

def init_db():
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            answer TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

class Agent:
    def __init__(self, client, agent_id, db_conn):
        self.client = client
        self.agent_id = agent_id
        self.db_conn = db_conn
        self.cursor = db_conn.cursor()

    def answer(self, mcq, scrambled_elements):
        memory = self.get_memory()
        memory_str = "\n".join(memory)
        
        prompt = f"""Please answer the following multiple-choice question:

{mcq}

Your answer (provide ONLY the letter of your choice, e.g., 'a' or 'b' or 'c' or 'd', nothing more):

Now, based on the following scrambled elements and your memory of previous questions, please provide your analysis or response:
{scrambled_elements}

Your memory of previous questions:
{memory_str}

Your response:"""

        message = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Extract only the letter answer using regex
        match = re.search(r'^([a-d])', response.lower().strip())
        if match:
            mcq_answer = match.group(1)
        else:
            mcq_answer = "Invalid"
        
        return mcq_answer, response

    def update_memory(self, question, answer, correct_answer):
        result = "correct" if answer.upper() == correct_answer.upper() else "incorrect"
        
        self.cursor.execute('''
            INSERT INTO questions (question, agent_id, answer, correct_answer, result)
            VALUES (?, ?, ?, ?, ?)
        ''', (question, self.agent_id, answer, correct_answer, result))
        
        self.db_conn.commit()

    def get_memory(self):
        self.cursor.execute('''
            SELECT question, answer, correct_answer, result
            FROM questions
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT 5
        ''', (self.agent_id,))
        
        memory = []
        for row in self.cursor.fetchall():
            question, answer, correct_answer, result = row
            memory_entry = f"Question: {question}\nYour answer: {answer}\nCorrect answer: {correct_answer}\nResult: {result}"
            memory.append(memory_entry)
        
        return memory

def create_agents(num_agents, client, db_conn):
    return [Agent(client, f"agent_{i}", db_conn) for i in range(num_agents)]

def get_all_responses(agents, elements, mcq, correct_answer):
    correct_count = 0
    for agent in agents:
        scrambled = scramble_sequence(elements)
        scrambled_str = json.dumps(scrambled, indent=2)
        mcq_answer, _ = agent.answer(mcq, scrambled_str)
        
        is_correct = mcq_answer.upper() == correct_answer.upper()
        agent.update_memory(mcq, mcq_answer, correct_answer)
        
        if is_correct:
            correct_count += 1
    
    return correct_count, len(agents)

def format_mcq(question_data):
    question = question_data["question"]
    options = question_data["options"]
    mcq = f"{question}\n" + "\n".join(f"{key}) {value}" for key, value in options.items())
    return mcq

if __name__ == "__main__":
    json_file_path = "init.json"
    questions_file_path = "question2.json"
    num_agents = 10

    # Initialize SQLite database
    db_conn = init_db()

    elements = load_json_file(json_file_path)
    questions = load_json_file(questions_file_path)
    
    try:
        claude_client = init_claude()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your ANTHROPIC_API_KEY environment variable.")
        exit(1)
        
    agents = create_agents(num_agents, claude_client, db_conn)

    question_accuracies = {i: [] for i in range(len(questions))}

    iteration = 1
    try:
        while True:
            print(f"\nIteration {iteration}")
            print("-" * 20)
            
            for i, question_data in enumerate(questions):
                mcq = format_mcq(question_data)
                correct_answer = question_data["answer"]
                
                correct_count, total_count = get_all_responses(agents, elements, mcq, correct_answer)
                accuracy = correct_count / total_count
                question_accuracies[i].append(accuracy)
                
                print(f"Question {i+1} - Accuracy: {accuracy:.2f}")
            
            print("\nAccuracy trend for each question:")
            for i, accuracies in question_accuracies.items():
                trend = " -> ".join(f"{acc:.2f}" for acc in accuracies)
                print(f"Question {i+1}: {trend}")
            
            continue_prompt = input("\nDo you want to run another iteration? (yes/no): ")
            if continue_prompt.lower() != 'yes':
                break
            
            iteration += 1
    finally:
        db_conn.close()