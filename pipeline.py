from agents import (
    build_search_agent,
    Read_agent,
    writer_chain,
    critic_chain
)


def run_pipeline(topic: str) -> dict:
    state = {}

    # -----------------------------
    # 1. SEARCH AGENT
    # -----------------------------
    print("\n========== SEARCHING ==========\n")

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about {topic}"
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print(state["search_results"])


    # -----------------------------
    # 2. READER / SCRAPER AGENT
    # -----------------------------
    print("\n========== READING SOURCES ==========\n")

    reader_agent = Read_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results for the topic "{topic}",
identify the most relevant URLs and scrape them for deeper information.

Search Results:
{state["search_results"]}
"""
            )
        ]
    })

    state["reader_results"] = reader_result["messages"][-1].content

    print(state["reader_results"])


    # -----------------------------
    # 3. WRITER
    # -----------------------------
    print("\n========== GENERATING REPORT ==========\n")

    writer_result = writer_chain.invoke({
        "topic": topic,
        "research": state["reader_results"]
    })

    state["report"] = writer_result

    print(state["report"])


    # -----------------------------
    # 4. CRITIC
    # -----------------------------
    print("\n========== CRITIC REVIEW ==========\n")

    critic_result = critic_chain.invoke({
        "report": state["report"]
    })

    state["critique"] = critic_result

    print(state["critique"])


    # -----------------------------
    # RETURN FINAL STATE
    # -----------------------------
    return state




if __name__ == "__main__":

    topic = input("Enter the research topic: ").strip()

    if not topic:
        print("Please enter a research topic.")
    else:
        result = run_pipeline(topic)