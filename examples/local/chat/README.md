# Multi-turn Conversation Example for Chat Applications

MII can manage multi-turn conversations, enabling users to easily create their own chat applications.
The scripts in this folder provide a complete example of a multi-turn conversation scenario.

## Starting the server

Starting the server for your chat application requires nothing special.
Just make sure that the model supports `text-generation` and is trained for conversations.

The example script uses [AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed](https://huggingface.co/AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed), which was trained using [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md).

```python
name = "AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed"
...
mii.deploy(task='text-generation', model=name, deployment_name="chat_example_deployment")
```

## Running multi-turn conversation

The client create a *session* to make MII recognize the session of the conversation.
`create_session` creates a new session with a given name.

```python
# You can set a session name
session_id = "chat_example_session"
# You need to call `create_session` before you start a multi-turn conversation session
generator.create_session(session_id)
```

The session ID is given as a keyword argument as shown below.
Besides this, you can call `query` function as well as the normal usage of MII inference.
Note that the prompt may need to be designed according to your model.

```python
query_template = "Human: {}\n Assistant: "
print("# Start a conversation session. Type 'q' to exit.")

while True:
    user_input = input("You: ")
    if user_input == "q":
        break
    result = generator.query({"query": query_template.format(user_input)},
                             session_id=session_id,
                             max_new_tokens=128)
    print(f"Bot: {result.response[0].replace('</s>', '')}")
```

Once the user finish the session, you need to call `destroy_session` to free the internal data for the session.

```python
generator.destroy_session(session_id)
```

The following is an example conversation:
```bash
$ python chat-client-example.py
# Start a conversation session. Type 'q' to exit.
You: Can you tell me about deep learning?
Bot:  Yes, it is a type of artificial intelligence that learns from data.  It can process large amounts of data quickly and accurately, and it can develop sophisticated models to analyze data.  Deep learning techniques are being applied to a wide variety of tasks, including image recognition, speech recognition, recommendation systems, and self-driving cars.
You: I want to start with a simple application.
Bot:  Yes, it is a good idea to start with a simple application.  It is possible to create simple neural networks that can learn and improve their performance over time.  The first step is to train the network by providing it with a large amount of data.  The performance of the network improves over time as more and more data is fed into it.
You: Can I use DeepSpeed?
Bot:  Yes, it is a good idea to use DeepSpeed for deep learning.  It provides high performance and efficiency for training neural networks.
You: Do I need a GPU machine?
Bot:  Yes, it is a good idea to use a GPU machine for deep learning.  It can speed up training by several orders of magnitude.
```
