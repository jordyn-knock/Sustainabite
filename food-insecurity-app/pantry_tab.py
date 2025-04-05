#this is the tab for the pantry and list columns
import streamlit as st

def render_pantry_tab():
    st.header("My Pantry List")
    col1, col2 = st.columns(2)
    #initialization...
    if "pantry" not in st.session_state:
        st.session_state.pantry = []
    if "grocery" not in st.session_state:
        st.session_state.grocery = []
    #done w initalization...
    with col1:
        st.subheader("My Pantry")
        new_item = st.text_input("Add to pantry", key="pantry_input")

        if st.button("Add to Pantry"):
            if new_item and new_item.lower() not in st.session_state.pantry:
                st.session_state.pantry.append(new_item.lower())

        if st.session_state.pantry:
            for i, item in enumerate(st.session_state.pantry):
                colA, colB = st.columns([4, 1])
                colA.write(f"- {item}")
                if colB.button("🗑️", key=f"remove_pantry_{i}"):
                    st.session_state.pantry.pop(i)
                    st.experimental_rerun()
        else:
            st.write("No pantry items yet.")

        if st.button("Clear Pantry"):
            st.session_state.pantry.clear()
            st.experimental_rerun()

    with col2:
        st.subheader("My Grocery List")
        grocery_item = st.text_input("Add to grocery list", key="grocery_input")

        if st.button("Add to Grocery List"):
            if grocery_item and grocery_item.lower() not in st.session_state.grocery:
                st.session_state.grocery.append(grocery_item.lower())

        if st.session_state.grocery:
            for i, item in enumerate(st.session_state.grocery):
                colA, colB = st.columns([4, 1])
                colA.write(f"- {item}")
                if colB.button("🗑️", key=f"remove_grocery_{i}"):
                    st.session_state.grocery.pop(i)
                    st.experimental_rerun()
        else:
            st.write("No grocery items yet.")

        if st.button("Clear Grocery List"):
            st.session_state.grocery.clear()
            st.experimental_rerun() #this is what forces the app to update, so do not remove it

    st.markdown("---")
    st.checkbox("🛍️ I'm willing to go to the grocery store, use items from My List", key="use_grocery")
