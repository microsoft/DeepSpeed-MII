# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

# Run `chat-server-example.py` before running this script
generator = mii.mii_query_handle("chat_example_deployment")

# You can set a session name
session_id = "chat_example_session"
# You need to call `create_session` before you start a multi-turn conversation session
generator.create_session(session_id)

print("# Start a conversation session. Type 'q' to exit.")
query_template = "Human: {}\n Assistant: "
while True:
    user_input = input("You: ")
    if user_input == "q":
        break

    # A session ID is given as a keyword argument
    result = generator.query({"query": query_template.format(user_input)},
                             session_id=session_id,
                             max_new_tokens=128)
    print(f"Bot: {result.response[0].replace('</s>', '').strip()}")

# You need to destroy the session after finishing the conversation
generator.destroy_session(session_id)
