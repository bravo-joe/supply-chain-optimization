# Beginning of app.py"
# Import necessary libraries and modules
from math import ceil
import yaml # Read config files 
import faicons as fa
import pandas as pd
from shiny import App, render, ui
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    # geom_boxplot,
    scale_x_discrete,
    labs,
    coord_flip,
    geom_histogram,
    theme,
    theme_minimal,
    element_text
)
from helper_funcs import query_tbl, extract_demand_df, extract_supply_df

# Config
with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

TRANSPORTATION_COSTS = query_tbl(
    CONFIG["server"]["user"],
    CONFIG["server"]["password"],
    CONFIG["server"]["host"],
    CONFIG["server"]["port"],
    CONFIG["database"]["name"],
    tbl_name="transportation_costs"
)
CONVERGENCE_CURVE = query_tbl(
    CONFIG["server"]["user"],
    CONFIG["server"]["password"],
    CONFIG["server"]["host"],
    CONFIG["server"]["port"],
    CONFIG["database"]["name"],
    tbl_name="convergence_history"
)

eda_df = pd.read_csv("./data/eda_df.csv")
# Demand
DEMAND = extract_demand_df(TRANSPORTATION_COSTS)
avg_demand = DEMAND['demand'].mean()
n_customers = len(DEMAND['demand'])
# Supply
SUPPLY = extract_supply_df(TRANSPORTATION_COSTS)
avg_supply = SUPPLY['production_capacity'].mean()
n_factories = len(SUPPLY['production_capacity'])

# Icons for value box
ICONS = {
    "clipboard": fa.icon_svg("clipboard"),
    "truck": fa.icon_svg("truck"),
    "chart": fa.icon_svg("chart-line"),
    "plant": fa.icon_svg("industry")
}

# The contents of the first 'page' is a navset with one panel
page1 = ui.navset_card_underline(
    ui.nav_panel(
        "Overview",
        ui.h3("Key Metrics"),
        ui.layout_column_wrap(  # Use layout_column_wrap for multiple value boxes
            ui.value_box(
                title="Number of Factories",
                value=ui.output_ui("number_plants"),
                showcase=ICONS['plant'],
                theme="primary"
            ),
            ui.value_box(
                title="Average Production Capacity (units)",
                value=ui.output_ui("production_capacity_mean"),
                showcase=ICONS['clipboard'],
                theme="info"
            ),
            ui.value_box(
                title="Number of Customers",
                value=ui.output_ui("number_customers"),
                showcase=ICONS['truck'],
                theme="success"
            ),
            ui.value_box(
                title="Average Demand (units)",
                value=ui.output_ui("demand_mean"),
                showcase=ICONS['chart'],
                theme="info"
            ),
        )
    ),
    ui.nav_panel(
        "Demand and Production Capacity",
        ui.output_plot("hist"),
        ui.input_select(
            id = "var",
            label = "Select variable",
            choices = [
                "demand",
                "production_capacity"
            ]
        )
    ),
    title = 'EDA'
)

page2 = ui.navset_card_underline(
    ui.nav_panel(
        "Convergence Curve",
        ui.output_plot('convergence_curve')
    ),
    title = "Ant Colony Optimization (ACO)"
)

app_ui = ui.page_navbar(
    ui.nav_spacer(), # Pushing the navbar items to the right
    ui.nav_panel(
        "Page 1",
        page1
    ),
    ui.nav_panel(
        "Page 2",
        # "This is a placeholder and second 'page'."
        page2
    ),
    # Main dashboard title
    title="Dashboard prototype",
)

# Define the server function
def server(
        input,
        output,
        session
):
    @render.ui
    def demand_mean():
        return ceil(avg_demand)
    
    @render.ui
    def production_capacity_mean():
        return ceil(avg_supply)
    
    @render.ui
    def number_customers():
        return n_customers
    
    @render.ui
    def number_plants():
        return n_factories

    @render.plot
    def hist():
        h = (
            ggplot(eda_df, aes(x=input.var()))\
            + geom_histogram(
                bins=10,
                fill='#5cb5d3',
                color='#040e05'
            )\
            + theme(axis_text_x=element_text(rotation=90,hjust=1))
        )
        return h
    
    @render.plot
    def convergence_curve():  # Create the convergence plot
        df_long = pd.melt
        df_long = pd.melt(
            CONVERGENCE_CURVE,
            id_vars=['iteration'],
            value_vars=['best_cost', 'avg_cost'],
            var_name='metric',
            value_name='cost'
        )
        cc = (
            ggplot(df_long, aes(x='iteration', y='cost', color='metric'))\
            + geom_line(size=1.2)\
            + geom_point(size=2, alpha=0.6)\
            + labs(
                title="ACO Cost Convergence Curve",
                x='Iteration',
                y='Cost',
                color='metric'
            )\
            + theme_minimal()\
            + theme(
                plot_title=element_text(
                    size=14,
                    face='bold'
                ),
                axis_title=element_text(size=11),
                figure_size=(10, 6)
            )
        )
        return cc
    
app = App(app_ui, server)

# End of app.py"