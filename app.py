# Beginning of app.py"

# Import necessary libraries and modules
import faicons as fa
import pandas as pd
from shiny import App, render, ui
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    scale_x_discrete,
    labs,
    coord_flip,
    geom_histogram,
    theme,
    element_text
)

# Locations of CSV files for the plots in dashboard
eda_df = pd.read_csv("./data/eda_df.csv")
avg_demand = 183.92
avg_cap = 446.67

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
                title="Total Sales",
                value="$1,234,567",
                # showcase=ui.bs_icon("cash-coin"),
                showcase=ICONS['plant'],
                theme="primary"
            ),
            ui.value_box(
                title="Customers",
                value="12,345",
                #showcase=ui.bs_icon("people"),
                showcase=ICONS['chart'],
                theme="success"
            ),
            ui.value_box(
                title="Average Demand (units)",
                value=ui.output_ui("mean_demand"),
                # showcase=ui.bs_icon("box-seam"),
                showcase=ICONS['truck'],
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
    # ui.nav_panel(
    #     "Customer Sample",
    #     ui.output_data_frame("sample_bxplt")
    # ),
    title = 'EDA',
)

app_ui = ui.page_navbar(
    # 1. Basic value box
    # ui.value_box(
    #     "Average Demand (units)",
    #     ui.output_ui("mean_demand"),
    #     showcase = ICONS['chart'],
    #     theme="primary"
    # ),
    ui.nav_spacer(), # Pushing the navbar items to the right
    ui.nav_panel(
        "Page 1",
        page1
    ),
    ui.nav_panel(
        "Page 2",
        "This is a placeholder and second 'page'."
    ),
    # Main dashboard title
    title="Dashboard prototype",
)

# ui.layout_columns(
#         # 1. Basic value box
#         ui.value_box(
#             "Average Demand (units)",
#             ui.output_ui("mean_demand"),
#             showcase = ICONS['chart'],
#             theme="primary"
#         ),
#         # 2. Average production capacity per plant
#         ui.value_box(
#             "Average production capacity (units)",
#             ui.output_ui("mean_production_capacity"),
#             showcase = ICONS['plant'],
#             theme = 'primary'
#         ),
#         fill=False
#     ), ui.page_navbar(


# Define the server function
def server(
        input,
        output,
        session
):
    @render.ui
    def mean_demand():
        return avg_demand
    
    @render.ui
    def mean_production_capacity():
        return avg_cap

    @render.plot
    def hist():
        # Plotting Logic
        h = ggplot(
            # input.var()
            eda_df
        )\
        + aes(
            x=input.var()
        )\
        + geom_histogram(
            bins=10,
            fill='#5cb5d3',
            color='#040e05'
        )\
        + theme(
            axis_text_x = element_text(rotation=90,hjust=1)
        )
        return h
    
app = App(app_ui, server)

# End of app.py"