import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from transformers import pipeline

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sentiment Analysis Dashboard"


pipelines = {
    "BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    "roBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "DistilBERT": pipeline("sentiment-analysis", model="distilbert-base-uncased"),
    "ALBERT": pipeline("sentiment-analysis", model="textattack/albert-base-v2-imdb"),
    "XLNet": pipeline("sentiment-analysis", model="xlnet-base-cased")
}

# App layout
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Sentiment Analysis", className="text-center my-4"), width=12)
    ),
    dbc.Row([ 
        dbc.Col([
            html.Label("Enter Text for Sentiment Analysis:", className="h5"),
            dcc.Textarea(
                id="input-text",
                placeholder="Type your text here...",
                style={"width": "100%", "height": "150px"},
            ),
        ], width=12),
    ], className="my-3"),
    dbc.Row([ 
        dbc.Col([
            html.Label("Choose a Sentiment Analysis Model:", className="h5"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": model, "value": model} for model in pipelines.keys()],
                placeholder="Select a model",
            ),
        ], width=12),
    ], className="my-3"),
    dbc.Row([ 
        dbc.Col(
            dbc.Button("Analyze Sentiment", id="analyze-btn", color="dark", className="mt-4 mb-4",style={"display": "block", "margin-left": "auto", "margin-right": "auto"}), 
            width=12
        )
    ]),
    dbc.Row([ 
        dbc.Col(
            dcc.Graph(id="sentiment-bar-graph", animate=True,style={"border": "2px solid #ddd", "padding": "20px", "border-radius": "10px"}), 
            width=12
        )
    ]),
    dbc.Row([ 
        dbc.Col(html.Div(id="sentiment-results"), width=12)  # Display sentiment results here
    ]),
    dbc.Row([ 
        dbc.Col(
            html.Div(id="sentiment-results", style={
                "background-color": "#f4f6f9",  # Same as background color
                "border": "1px solid #ddd",  # Light gray border
                "padding": "20px",
                "border-radius": "10px",
                "text-align": "center",
                "margin-top": "20px",
                "font-family": "'Roboto', sans-serif",  # Change font to Roboto
                "font-size": "18px",  # Adjust font size as needed
            }), 
            width=12
        )
    ]),
], fluid=True, style={"background-color": "#f4f6f9", "min-height": "100vh"})


@app.callback(
    [Output("sentiment-bar-graph", "figure"),
     Output("sentiment-results", "children")],
    [Input("analyze-btn", "n_clicks")],
    [State("input-text", "value"), State("model-dropdown", "value")]
)
def analyze_sentiment(n_clicks, text, model_name):
    if not n_clicks or not text or not model_name:
        return go.Figure(), ""

    try:
        selected_pipeline = pipelines[model_name]
        results = selected_pipeline(text)
        print("Pipeline Results:", results)  

        # Define label mappings for each model
        label_mappings = {
            "BERT": {
                "5 stars": "positive", "4 stars": "positive",
                "3 stars": "neutral",
                "2 stars": "negative", "1 star": "negative",
            },
            "roBERTa": {
                "LABEL_2": "positive",
                "LABEL_1": "neutral",
                "LABEL_0": "negative",
            },
            "DistilBERT": {
                "LABEL_1": "positive",  
                "LABEL_0": "negative", 
            },
            "ALBERT": {
                "LABEL_1": "positive",
                "LABEL_0": "negative",
            },
            "XLNet": {
                "LABEL_1": "positive",
                "LABEL_0": "negative",
            },
        }

        # Get the mapping for the selected model
        model_mapping = label_mappings.get(model_name, {})
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

        for result in results:
            label = result['label']
            score = result['score']
            mapped_label = model_mapping.get(label)
            if mapped_label:
                sentiments[mapped_label] += 1
                sentiment_scores[mapped_label] += score
            else:
                print(f"Warning: Unmapped label '{label}' for model '{model_name}'")

        # Calculate the percentage confidence score for each sentiment
        total_score = sum(sentiment_scores.values())
        sentiment_percentage = {key: (score / total_score) * 100 if total_score > 0 else 0
                               for key, score in sentiment_scores.items()}

        # Create bar graph
        figure = go.Figure(
            data=[go.Bar(
                x=list(sentiments.keys()),
                y=list(sentiments.values()),
                marker_color=["#494F55", "#494F55", "#494F55"],
                text=[f"{sentiment_percentage[key]:.2f}%" for key in sentiments.keys()],
                textposition="outside",
            )],
            layout=go.Layout(
                title="Sentiment Analysis Results",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                template="plotly_white",
                updatemenus=[{
                    'buttons': [{
                        'args': [None, {'frame': {'duration': 2000, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate',
                    }],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }],  
            ),
            frames=[
                go.Frame(
                    data=[go.Bar(
                        x=list(sentiments.keys()),
                        y=[0] * len(sentiments),  # Initially set to zero
                        marker_color=["#494F55", "#494F55", "#494F55"],
                    )],
                    name="start"
                ),
                go.Frame(
                    data=[go.Bar(
                        x=list(sentiments.keys()),
                        y=list(sentiments.values()),  # Values appear gradually
                        marker_color=["#494F55", "#494F55", "#494F55"],
                    )],
                    name="end"
                ),
            ]
        )

        # Prepare sentiment results text with sentiment score and confidence score
        sentiment_results = [
            html.Div(f"Positive Sentiment: {sentiments['positive']} | Confidence: {sentiment_percentage['positive']:.2f}%"),
            html.Div(f"Neutral Sentiment: {sentiments['neutral']} | Confidence: {sentiment_percentage['neutral']:.2f}%"),
            html.Div(f"Negative Sentiment: {sentiments['negative']} | Confidence: {sentiment_percentage['negative']:.2f}%")
        ]

        return figure, sentiment_results

    except Exception as e:
        print(f"Error: {e}")
        return go.Figure(), "Error occurred during sentiment analysis."


# Run app
if __name__ == "__main__":
    app.run_server(debug=True)