from dotenv import load_dotenv





load_dotenv()

from graph.graph import app


# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Working?","Yesssirr")
    print(app.invoke({
        "question": "What is an agent memory?"
    }))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
