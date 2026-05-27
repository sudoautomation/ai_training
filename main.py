# main.py

from app.agents.router import route_query


while True:

    query = input("\nUser: ")

    if query.lower() == "exit":
        break

    response = route_query(query)

    print("\n====================")
    print(response)