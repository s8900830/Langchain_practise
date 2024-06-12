from ollama import Client

CLIENT = Client(host='http://127.0.0.1:11434')
MODEL_NAME='llama3'
def post(message = ''):

    response = CLIENT.chat(
        model=f'{MODEL_NAME}', 
        messages=[{'role': 'user','content': f'{message}'}],
        stream=True
    )
    ans =''
    for chunk in response:
        ans += chunk['message']['content']
    return ans

if __name__=="__main__":
    msg = input()
    print(post(msg))