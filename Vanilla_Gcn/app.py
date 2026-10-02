"""
Streamlit web page that displays output.txt of the GCN fake-account detector.
Run:  pip install streamlit
      streamlit run app.py
"""
import os
import pandas as pd
import streamlit as st

FILE = "output.txt"

st.set_page_config(page_title="GCN Fake Account Detection", page_icon="🕵️", layout="wide")


def read_text(path):
    """Read output.txt safely (handles PowerShell's UTF-8 BOM / UTF-16)."""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            if "\x00" not in text:
                return text
        except UnicodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def parse(text):
    epochs, rows = [], []
    for line in text.splitlines():
        p = line.split()
        if line.startswith("Epoch"):
            epochs.append({"Epoch": int(p[1]),
                           "Confidence on labeled": float(p[6]),
                           "Accuracy on unlabeled": float(p[-1])})
        elif len(p) >= 4 and p[2] in ("REAL", "FAKE") and p[3] in ("REAL", "FAKE"):
            rows.append({"Account": p[0], "P(fake)": float(p[1]),
                         "Predicted": p[2], "Actual": p[3],
                         "Labeled by human": "Yes" if "(labeled)" in line else "No"})
    return pd.DataFrame(epochs), pd.DataFrame(rows)


# ------------------------------------------------------------------
st.title("🕵️ Fake Account Detection using GCN")
st.caption("Vanilla Graph Convolutional Network, pure Python | Algorithm Design assignment")

if not os.path.exists(FILE):
    st.error(f"'{FILE}' not found. Put it in the same folder as app.py "
             "(create it with: python fake_account_gcn.py | Out-File -Encoding utf8 output.txt)")
    st.stop()

text = read_text(FILE)
epochs_df, results_df = parse(text)

if results_df.empty:
    st.warning("output.txt was found but no results could be read. Re-create it and try again.")
    st.stop()

correct = int((results_df["Predicted"] == results_df["Actual"]).sum())
total = len(results_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts", total)
c2.metric("Predicted FAKE", int((results_df["Predicted"] == "FAKE").sum()))
c3.metric("Predicted REAL", int((results_df["Predicted"] == "REAL").sum()))
c4.metric("Correct", f"{correct}/{total}")

tab1, tab2, tab3, tab4 = st.tabs(["📄 Output file", "📊 Results", "📈 Training", "🔎 Check an account"])

with tab1:
    st.subheader("Contents of output.txt")
    st.code(text, language="text")
    st.download_button("Download output.txt", text, file_name="output.txt")

with tab2:
    st.subheader("Probability that each account is FAKE")
    st.bar_chart(results_df.set_index("Account")["P(fake)"])

    def color_row(row):
        color = "#fde8ea" if row["Predicted"] == "FAKE" else "#e3f4f1"
        return [f"background-color: {color}; color: black"] * len(row)

    st.dataframe(results_df.style.apply(color_row, axis=1).format({"P(fake)": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Training progress")
    if epochs_df.empty:
        st.info("No epoch lines found in output.txt")
    else:
        st.line_chart(epochs_df.set_index("Epoch"))
        st.dataframe(epochs_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Look up one account")
    name = st.selectbox("Choose an account", results_df["Account"].tolist())
    r = results_df[results_df["Account"] == name].iloc[0]
    st.progress(float(r["P(fake)"]), text=f"P(fake) = {r['P(fake)']:.3f}")
    if r["Predicted"] == "FAKE":
        st.error(f"{name} is predicted FAKE (actual: {r['Actual']})")
    else:
        st.success(f"{name} is predicted REAL (actual: {r['Actual']})")
    st.write(f"Labeled by a human before training: **{r['Labeled by human']}**")