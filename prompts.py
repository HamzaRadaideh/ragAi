context = """Purpose: The primary role of this agent is to assist the users in any task they need and spicially coding """

code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code if there was code available, 
                            also come up with a valid filename this could be saved as that doesn't contain special characters. 
                            Here is the response: {response}. You should parse this in the following JSON format:"""