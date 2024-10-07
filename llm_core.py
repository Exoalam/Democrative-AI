import json
import random
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

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

Your answer (provide only the letter of your choice, nothing more and no explanation):

Now, based on the following scrambled elements and your memory of previous questions, please provide your analysis or response:
{scrambled_elements}

Your memory of previous questions with correct and incorrect answers. Learn from your mistakes and next time you see same question try to answer correctly:
{memory}

Your response:"""
)

class Agent:
    def __init__(self, llm, prompt):
        self.chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        self.memory = []
        self.correct_answers = 0
        self.total_questions = 0

    def answer(self, mcq, scrambled_elements):
        memory_str = "\n".join(self.memory)
        response = self.chain.run(mcq=mcq, scrambled_elements=scrambled_elements, memory=memory_str)
        mcq_answer = response.split()[0]
        return mcq_answer, response

    def update_memory(self, question, answer, correct_answer):
        result = "correct" if answer.upper() == correct_answer.upper() else "incorrect"
        memory_entry = f"Question: {question}\nYour answer: {answer}\nCorrect answer: {correct_answer}\nResult: {result}"
        self.memory.append(memory_entry)
        if len(self.memory) > 500:  # Keep only the last 5 memories
            self.memory.pop(0)

    def update_score(self, correct):
        self.total_questions += 1
        if correct:
            self.correct_answers += 1

def create_agents(num_agents, llm):
    return [Agent(llm, prompt_template) for _ in range(num_agents)]

def get_all_responses(agents, elements, mcq, correct_answer):
    responses = []
    for i, agent in enumerate(agents, 1):
        scrambled = scramble_sequence(elements)
        scrambled_str = json.dumps(scrambled, indent=2)
        mcq_answer, full_response = agent.answer(mcq, scrambled_str)
        
        is_correct = mcq_answer.upper() == correct_answer.upper()
        agent.update_memory(mcq, mcq_answer, correct_answer)
        agent.update_score(is_correct)
        
        accuracy = agent.correct_answers / agent.total_questions if agent.total_questions > 0 else 0
        responses.append(f"Agent {i}:\nMCQ Response: {mcq_answer}\nCorrect: {is_correct}\nAccuracy: {accuracy:.2f}")
    
    return responses

def get_mcq_from_user():
    print("Enter your multiple-choice question:")
    question = input("Question: ")
    options = []
    print("Enter the options (one per line). Type 'done' when finished:")
    while True:
        option = input()
        if option.lower() == 'done':
            break
        options.append(option)
    
    mcq = f"{question}\n" + "\n".join(f"{chr(65+i)}) {option}" for i, option in enumerate(options))
    correct_answer = input("Enter the correct answer (A, B, C, etc.): ")
    return mcq, correct_answer

if __name__ == "__main__":
    json_file_path = "init.json"
    ollama_base_url = "http://vrworkstation.atr.cs.kent.edu:11434"
    ollama_model = "llama3.2:3b"
    num_agents = 10

    elements = load_json_file(json_file_path)
    llm = init_ollama(ollama_base_url, ollama_model)
    agents = create_agents(num_agents, llm)

    while True:
        # Get MCQ and correct answer from user
        mcq, correct_answer = get_mcq_from_user()

        all_responses = get_all_responses(agents, elements, mcq, correct_answer)
        
        print(f"\nResponses from {num_agents} agents:\n")
        for response in all_responses:
            print(response)
            print("-" * 50)

        continue_prompt = input("Do you want to ask another question? (yes/no): ")
        if continue_prompt.lower() != 'yes':
            break

    print("\nFinal Agent Accuracies:")
    for i, agent in enumerate(agents, 1):
        accuracy = agent.correct_answers / agent.total_questions if agent.total_questions > 0 else 0
        print(f"Agent {i}: {accuracy:.2f}")