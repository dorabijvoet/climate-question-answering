def set_defaults(state):
    print("---- Setting defaults ----")
    
    if not state["audience"] or state["audience"] is None:
        state.update({"audience": "experts"})   

    if not state["sources_input"] or state["sources_input"] is None:
        state.update({"sources_input": ["auto"]})  
                                 
    return state