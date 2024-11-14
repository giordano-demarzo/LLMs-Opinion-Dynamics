import openai
import random
import time
import threading
import replicate
import anthropic
import string 

#Anthropic key
client = anthropic.Anthropic(
    api_key="YOUR_API_KEY",
)


#OpenAI key
openai.api_key = "YOUR_API_KEY" 


def create_chatbot(query, model, T=0.7):
    """ gpt-3.5-turbo-1106 
        gpt-3.5-turbo-0125
        gpt-4-0125-preview MOST SIMULATIONS
        gpt-4-turbo-2024-04-09 BEST GPT4 TURBO
        gpt-4-0613	 OLD GPT4
        gpt-4o-2024-05-13 
        gpt-4o-mini-2024-07-18
        o1-preview-2024-09-12	
    """
    query = {"role": "user", "content": query},
    chatbot = openai.ChatCompletion.create(
        model=model,
        messages=query,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=T,
    )

    return chatbot['choices'][0]['message']['content']

def create_chatbot_llama(query, model, T=0.7, verbose=False):
    
    input = {
        "temperature": T,
        "prompt": query,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    }
    response = ""
    
    model = "meta/" + model
    # Stream and collect events into the response string until an empty string is received
    for event in replicate.stream(
        model,
        input=input
    ):
        if verbose:
            print(str(event), end="")
    # If the event is an empty string, stop the loop
        if event == "":
            break
        # Otherwise, concatenate the event to the response string
        response += str(event)


    return response

def create_chatbot_claude(query, model, T=0.7):
    """
        claude-3-haiku-20240307  
        claude-3-sonnet-20240229
        claude-3-opus-20240229
        claude-3-5-sonnet-20240620
    """
    query = {"role": "user", "content": query},
    chatbot = client.messages.create(
    model=model,
    max_tokens=100,
    temperature=T,
    messages=query
    )

    return chatbot.content[0].text


def run_with_timeout(func, args=(), timeout_duration=25, default=None):
    """Runs the given function with the given args.
       If the function takes more than timeout_duration seconds, 
       default is returned instead."""
    
    class InterruptableThread(threading.Thread):
        def __init__(self, func, args):
            threading.Thread.__init__(self)
            self.func = func
            self.args = args
            self.result = default

        def run(self):
            try:
                self.result = self.func(*self.args)
            except Exception as e:
                self.result = default

    it = InterruptableThread(func, args)
    it.start()
    it.join(timeout_duration)
    
    # If thread is still alive after join(timeout), it's running for too long
    if it.is_alive():
        print('timeout')
        return default  # Return default value
    
    return it.result

def generate_unique_random_strings(length, num_strings):
    """Generate a list of unique random strings."""
    generated_strings = set()
    
    # Combine ASCII letters, digits, and punctuation to form the character set
    chars = string.ascii_letters + string.digits #+ string.punctuation
    
    # Continue generating until we have the desired number of unique strings
    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)

def synchronized_shuffle(a, b):
    # Ensure the lists are of the same length
    assert len(a) == len(b)
    
    # Create a list of indices
    indices = list(range(len(a)))
    
    # Shuffle the indices
    random.shuffle(indices)
    
    # Rearrange the lists
    a_shuffled = [a[i] for i in indices]
    b_shuffled = [b[i] for i in indices]
    
    return a_shuffled, b_shuffled

#%%OPINION SIMULATION SINGLE AGENT
import numpy as np
import traceback
import string 



# model = "gpt-4-turbo-2024-04-09"
model = "meta-llama-3-70b-instruct"

T = 0.2
N = 50
list_N = ((N/50)*np.array([48, 40, 32, 30, 27, 23, 20, 18, 10, 2])).tolist()
list_N = [int(np.round(x)) for x in list_N]


N_sim = 200
opinion_names_initial = ['k', 'z']

for Na in list_N:
    Nb = N - Na
    m0 = (Na-Nb)/N
    print(Na, Nb, m0)
    
    counts_opinion = [0 , 0]
    
    for i in range(N_sim):
        print(i)
        print(counts_opinion)
        list_names = generate_unique_random_strings(3, N) 
        r = random.random()
        if r<=0.5:
            opinion_names = [opinion_names_initial[0], opinion_names_initial[1]]
        else:
            opinion_names = [opinion_names_initial[1], opinion_names_initial[0]]
        opinions = [opinion_names[0] for i in range(Na)] + [opinion_names[1] for i in range(Nb)]
    
        #PROMPT 1    
        string_query = """Below you can see the list of all your friends toghether with the opinion they support.
        You must reply with the opinion you want to support.
        The opinion must be reported between square brakets."""
        
        #PROMPT 2
        # string_query = f"""Below you can see a list of people toghether with the opinion they support.
        #                   You must reply with the opinion you want to support.
        #                   The opinion must be reported between square brakets."""
        
        #PROMPT 3
        # string_query = f"""The list below contains people along with the opinions they endorse.
        #            Please respond with the opinion you'd like to support.
        #            Be sure to enclose the opinion in square brackets."""
        
        #PROMPT 4
        # string_query = f"""You recently subscribed to a social network. 
        #                   Below you can see the list of all your friends toghether with the group they joined on the social network.
        #                   You must reply with the name of the group you want to join.
        #                   The name of the group must be reported between square brakets."""
        
        #PROMPT 5   
        # string_query = f"""You recently subscribed to a social network. 
        # Below you can see the list of all your friends toghether with the opinion they support.
        # You must reply with the opinion you want to support.
        # The opinion must be reported between square brakets."""
        
        #PROMPT MAJORITY   
        # string_query = f"""Below you can see the list of all your friends toghether with the opinion they support.
        #                       You must reply with the opinion supported by the majority.
        #                       The opinion must be reported between square brakets."""
        
    
        temp_names = list_names[:]
        temp_traits = opinions[:]
        temp_traits = [": "+str(x)+"\n" for x in temp_traits]
        temp_names, temp_traits = synchronized_shuffle(temp_names, temp_traits)
        string_personalized = f"""{"".join([x + temp_traits[i] for i, x in enumerate(temp_names)])}. 
        Reply only with the opinion you want to support, between square brackets."""
        query = string_query + '\n' + string_personalized
        check = 0 
        counts = 0
        print("\n")
        while check == 0 and counts<10:
            #print(counts)
            counts = counts + 1
            try:
                response = run_with_timeout(create_chatbot, (query, model, T, )) #change to use a different API 
                print(response)
                if "[" in response and "]" in response:
                    y = response.partition("[")[2].partition("]")[0]
                    print(y)
                    if opinion_names[0] in y:
                        print('A')
                        counts_opinion[0] = counts_opinion[0] + 1
                    elif opinion_names[1] in y:
                        print('B')
                        counts_opinion[1] = counts_opinion[1] + 1
                check = 1
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()                
                pass
    with open(f'transition_prob_{N}_{T}_{model}_{opinion_names_initial[0]}_{opinion_names_initial[1]}.txt', 'a') as file:
        # Write each item on a new line
        file.write(f"{m0},{counts_opinion[0]/N_sim}\n")

#%%OPINION SIMULATION MULTIPLE AGENT MEAN FIELD 
import random 
import numpy as np
import traceback


def simulation(Na, N, n_sim, t_max):
    Nb = N - Na
    m = (2*Na-N)/N
    i = 0
    
            
    while abs(m)<1 and i<t_max:
        m = (2*Na-N)/N
        print(i, m)
        with open(f'magnetization_{n_sim}_{N}_{T}_{m0}_{model}.txt', 'a') as file:
            file.write(f"{i},{m}\n")
        if abs(m) == 1:
            print('Consensus')
            break
        list_names = generate_unique_random_strings(2, N) 

        r = random.random()
        if r<=0.5:
            opinion_names = ['k', 'z']
        else:
            opinion_names = ['z', 'k']
        opinions = [opinion_names[0] for i in range(Na)] + [opinion_names[1] for i in range(Nb)]
        random.shuffle(opinions)
        
        idx = random.randint(0, N-1)
        selected_opinion = opinions[idx]
        
        string_query = """Below you can see the list of all your friends toghether with the opinion they support.
        You must reply with the opinion you want to support.
        The opinion must be reported between square brakets."""
        
        temp_names = list_names[:]
        temp_traits = opinions[:]
        temp_traits.pop(idx)
        temp_names.pop(idx)
        temp_traits = [": "+str(x)+"\n" for x in temp_traits]
        temp_names, temp_traits = synchronized_shuffle(temp_names, temp_traits)
        string_personalized = f"""{"".join([x + temp_traits[i] for i, x in enumerate(temp_names)])}. 
        Reply only with the opinion you want to support, between square brakets."""
        query = string_query + '\n' + string_personalized
        check = 0 
        counts = 0
        while check == 0:
            counts = counts + 1
            try:
                response = run_with_timeout(create_chatbot_llama, (query, model, T,)) #change to use a different API
                if "[" in response and "]" in response:
                    y = response.partition("[")[2].partition("]")[0]
                    if opinion_names[0] in y:
                        counts_opinion[0] = counts_opinion[0] + 1
                        check = 1
                        i = i + 1
                        if selected_opinion==opinion_names[1]:
                            Na = Na + 1
                            Nb = Nb - 1
                    elif opinion_names[1] in y:
                        counts_opinion[1] = counts_opinion[1] + 1
                        check = 1
                        i = i + 1
                        if selected_opinion==opinion_names[0]:
                            Na = Na - 1
                            Nb = Nb + 1
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()  
                pass
    return m, i


N = 50
N_sim = 20
T = 0.2
# model = "gpt-4o-2024-05-13"
model = "meta-llama-3-70b-instruct"


Na = int(N/2)
Nb = N - Na
N = Na + Nb
m0 = (2*Na-N)/N
t_max = 10*N

counts_opinion = [0 , 0]
counts_consensus = [0, 0]
all_times = []
all_m = []
for i in range(N_sim):
    m, t_cons = simulation(Na, N, i, t_max)
    all_times.append(t_cons)
    print(t_cons, m)
    all_m.append(m)
    if m<0:
        counts_consensus[1] = counts_consensus[1] + 1
    elif m>0:
        counts_consensus[0] = counts_consensus[0] + 1
    print(counts_consensus)

with open(f'list_times_{N}_{T}_{m0}_{model}_MF.txt', 'a') as file:
    # Write each item on a new line
    for item in all_times:
        file.write(f"{item}\n")
    
with open(f'list_magnetization_{N}_{T}_{m0}_{model}_MF.txt', 'a') as file:
    # Write each item on a new line
    for item in all_m:
        file.write(f"{item}\n")
        
