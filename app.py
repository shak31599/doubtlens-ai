"""Hugging Face Spaces entrypoint. No external API credentials are required."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import gradio as gr
import joblib
import pandas as pd
import plotly.graph_objects as go
from functools import lru_cache
from doubtlens.core import load,predict,retrieve,influential_terms,CATEGORIES

@lru_cache(maxsize=1)
def resources():
    try:
        models,metrics=load(ROOT)
        index=joblib.load(ROOT/'artifacts/retrieval.joblib')
        return models,metrics,index,None
    except Exception as exc:
        return None,None,None,f'Model artifacts unavailable: {exc}. Run `python scripts/train.py`.'

def classify(question,model_name):
    models,metrics,index,error=resources()
    if error:return error,{},'',pd.DataFrame()
    try:
        chosen=metrics['recommended'] if model_name=='Auto (validation winner)' else ('baseline' if model_name=='Word TF-IDF baseline' else 'character')
        result=predict(question,models[chosen]); score=result['confidence']
        status='⚠️ Uncertain prediction' if result['uncertain'] else 'Model prediction (not a verified answer)'
        top=f'### {result["category"]}\n**{status}** · Score: **{score:.1%}** · `{chosen}` model'
        explanation=influential_terms(question,models[chosen]); expl=', '.join(f'`{term}` ({value:.3f})' for term,value in explanation) or 'No positive token contributions for this example.'
        return top,result['probabilities'],'Scores are not necessarily calibrated; trained on synthetic examples only.\n\n**Influential TF-IDF features** (associations, not causation): '+expl,pd.DataFrame([{'model':n,'predicted_category':predict(question,m)['category'],'top_score':round(predict(question,m)['confidence'],3)} for n,m in models.items()])
    except ValueError as exc:return str(exc),{},'',pd.DataFrame()

def search(question,threshold):
    _,_,index,error=resources()
    if error:return pd.DataFrame(),error
    try:
        rows=retrieve(question,index,threshold=threshold)
        if not rows:return pd.DataFrame(),'No similar questions above the chosen cosine threshold.'
        return pd.DataFrame(rows),'Similarity is cosine similarity, not a probability. All examples are synthetic; offline fallback uses TF-IDF, not sentence embeddings.'
    except ValueError as exc:return pd.DataFrame(),str(exc)

def dashboard(name):
    _,metrics,_,error=resources()
    if error:return error,None,pd.DataFrame(),pd.DataFrame()
    m=metrics[name]
    figure=go.Figure(data=go.Heatmap(z=m['confusion_matrix'],x=CATEGORIES,y=CATEGORIES,colorscale='Blues',text=m['confusion_matrix'],texttemplate='%{text}'))
    figure.update_layout(title=f'{name.title()} confusion matrix — held-out synthetic test',xaxis_title='Predicted',yaxis_title='Actual',height=460)
    class_table=pd.DataFrame([dict(category=c,**values) for c,values in m['per_class'].items()])
    comparison=pd.DataFrame([{'model':n,'accuracy':metrics[n]['accuracy'],'macro_f1':metrics[n]['macro_f1'],'weighted_f1':metrics[n]['weighted_f1'],'validation_macro_f1':metrics[n]['validation_macro_f1'],'latency_ms_per_question':metrics[n]['inference_ms_per_question']} for n in ('baseline','character')])
    return f'**Accuracy:** {m["accuracy"]:.3f} · **Macro F1:** {m["macro_f1"]:.3f} · **Weighted F1:** {m["weighted_f1"]:.3f}  \n**Recommended:** {metrics["recommended"]} (selected with validation data)  \n**Important:** metrics are from original synthetic examples and do not establish real-world performance.',figure,class_table,comparison

def error_analysis(name):
    _,metrics,_,err=resources()
    if err:return pd.DataFrame(),err
    examples=metrics[name]['errors']
    return pd.DataFrame(examples,columns=['question_text','actual','predicted']),f'{len(examples)} misclassified questions in held-out synthetic test set. Do not use these errors to tune the reported model.'

CSS='''body{font-family:Inter,system-ui,sans-serif!important} .gradio-container{max-width:1150px!important} #hero{background:linear-gradient(130deg,#0f172a,#1e3a8a);padding:30px 32px;border-radius:18px;color:#fff} #hero h1,#hero p{color:#fff} .panel{border-radius:14px}'''
with gr.Blocks(title='DoubtLens AI — Intelligent Student Question Analysis',css=CSS,theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🔎 DoubtLens AI\n**Classify technical doubts. Discover related questions. Understand model performance.**',elem_id='hero')
    gr.Markdown('⚠️ **Research demonstration:** trained and evaluated with a small original synthetic dataset; outputs are estimates, not verified factual answers. Word/character TF-IDF models operate offline. A sentence-transformer extension is optional.')
    with gr.Tabs():
        with gr.Tab('🧠 Question classifier'):
            question=gr.Textbox(label='Your technical question',placeholder='Why does Dijkstra’s algorithm fail with negative edge weights?',lines=4)
            selector=gr.Radio(['Auto (validation winner)','Word TF-IDF baseline','Character TF-IDF alternative'],value='Auto (validation winner)',label='Classifier')
            submit=gr.Button('Analyze question',variant='primary')
            headline=gr.Markdown();probabilities=gr.Label(label='Scores by category');explanation=gr.Markdown();comparison=gr.Dataframe(label='Side-by-side prediction comparison')
            submit.click(classify,inputs=[question,selector],outputs=[headline,probabilities,explanation,comparison])
            gr.Examples(['Why does binary search require sorted input?','How can I fix a CORS preflight error in my React app?','What causes overfitting when training a neural network?','How do database transactions prevent dirty reads?'],inputs=question)
        with gr.Tab('🔍 Similar questions'):
            search_query=gr.Textbox(label='Find questions related to',lines=3)
            threshold=gr.Slider(0,1,value=.08,step=.01,label='Minimum cosine similarity')
            search_button=gr.Button('Search related questions',variant='primary')
            search_rows=gr.Dataframe(label='Most similar training questions');search_status=gr.Markdown()
            search_button.click(search,inputs=[search_query,threshold],outputs=[search_rows,search_status])
        with gr.Tab('📊 Model performance'):
            model=gr.Radio(['baseline','character'],value='baseline',label='Model')
            refresh=gr.Button('Load evaluation results')
            overview=gr.Markdown();cm=gr.Plot();perclass=gr.Dataframe(label='Per-class metrics');comp=gr.Dataframe(label='Model comparison')
            refresh.click(dashboard,inputs=model,outputs=[overview,cm,perclass,comp])
            model.change(dashboard,inputs=model,outputs=[overview,cm,perclass,comp])
        with gr.Tab('🧩 Error analysis'):
            error_model=gr.Radio(['baseline','character'],value='baseline',label='Model')
            error_button=gr.Button('View real evaluation errors');errors=gr.Dataframe();errnote=gr.Markdown()
            error_button.click(error_analysis,inputs=error_model,outputs=[errors,errnote])
        with gr.Tab('📚 Methodology and limitations'):
            gr.Markdown('''### How it works\n1. A **word TF-IDF + Logistic Regression** baseline and a **character n-gram TF-IDF + Logistic Regression** alternative train on original synthetic programming questions.\n2. A fixed, stratified 60/20/20 train/validation/test split prevents exact duplicates from crossing splits. The winner is selected on validation only, then evaluated on held-out test questions.\n3. Question search uses **cosine similarity over a TF-IDF training-question index** as a reliable CPU/offline fallback. A separate optional Sentence Transformers module supports true semantic embeddings when its weights are accessible.\n4. Predictions are not ground truth. No human-vetted real-world benchmark, calibration study, or user-traffic evaluation was performed. Synthetic question templates create shared patterns across splits, so test performance may be optimistic.\n5. Scores are **not necessarily calibrated**. Input can be outside the supported categories. Explanations are linear-model token associations, not causal explanations.\n\n**Reproduce:** run `python scripts/train.py` then `pytest -q`. Refer to dataset card, model card, and source repository documentation.''')
if __name__=='__main__':demo.launch(server_name='0.0.0.0',server_port=7860)
