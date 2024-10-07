import json
import random
import re
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pymongo import MongoClient

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

def init_ollama(base_url, model="llama3.2:3b"):
    return Ollama(base_url=base_url, model=model)

prompt_template = PromptTemplate(
    input_variables=["mcq", "scrambled_elements", "memory"],
    template="""Please answer the following multiple-choice question:

{mcq}

Your answer (provide ONLY the letter of your choice, e.g., 'a' or 'b' or 'c' or 'd', nothing more):

Now, based on the following scrambled elements and your memory of previous questions, please provide your analysis or response:
{scrambled_elements}

Your memory of previous questions:
{memory}

Your response:"""
)

class Agent:
    def __init__(self, llm, prompt, agent_id, db):
        self.chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        self.agent_id = agent_id
        self.db = db

    def answer(self, mcq, scrambled_elements):
        memory = self.get_memory()
        memory_str = "\n".join(memory)
        response = self.chain.run(mcq=mcq, scrambled_elements=scrambled_elements, memory=memory_str)
        
        # Extract only the letter answer using regex
        match = re.search(r'^([a-d])', response.lower().strip())
        if match:
            mcq_answer = match.group(1)
        else:
            mcq_answer = "Invalid"  # or some default value
        
        return mcq_answer, response

    def update_memory(self, question, answer, correct_answer):
        result = "correct" if answer.upper() == correct_answer.upper() else "incorrect"
        
        existing_question = self.db.questions.find_one({"question": question})
        
        if existing_question:
            self.db.questions.update_one(
                {"_id": existing_question["_id"]},
                {"$set": {f"agents.{self.agent_id}": {
                    "answer": answer,
                    "correct_answer": correct_answer,
                    "result": result
                }}}
            )
        else:
            self.db.questions.insert_one({
                "question": question,
                "agents": {
                    self.agent_id: {
                        "answer": answer,
                        "correct_answer": correct_answer,
                        "result": result
                    }
                }
            })

    def get_memory(self):
        questions = self.db.questions.find(
            {f"agents.{self.agent_id}": {"$exists": True}},
            sort=[("_id", -1)],
            limit=5
        )
        
        memory = []
        for q in questions:
            agent_data = q["agents"][self.agent_id]
            memory_entry = f"Question: {q['question']}\nYour answer: {agent_data['answer']}\nCorrect answer: {agent_data['correct_answer']}\nResult: {agent_data['result']}"
            memory.append(memory_entry)
        
        return memory

def create_agents(num_agents, llm, db):
    return [Agent(llm, prompt_template, f"agent_{i}", db) for i in range(num_agents)]

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
    ollama_base_url = "http://vrworkstation.atr.cs.kent.edu:11434"
    ollama_model = "llama3.2:3b"
    num_agents = 10

    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["agent_memory_db"]

    elements = load_json_file(json_file_path)
    questions = load_json_file(questions_file_path)
    llm = init_ollama(ollama_base_url, ollama_model)
    agents = create_agents(num_agents, llm, db)

    question_accuracies = {i: [] for i in range(len(questions))}

    iteration = 1
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

    mongo_client.close()