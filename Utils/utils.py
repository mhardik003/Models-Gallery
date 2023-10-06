import streamlit as st
import gc
def clear_ada_cache(model):
    """
    Function to clear cache in ada on task change in different models
    """
    model = None
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()
    # del model
    # gc.disable()
    return model
