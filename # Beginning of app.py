# Beginning of app.py"

# Import necessary libraries and modules
from shiny import App, render, ui

# Locations of CSV files for the plots in dashboard
demand = "./data/demand.csv"
prod_cap = "./data/production_capacity.csv"
avg_demand = 183.92


# The contents of the first 'page' is a navset with one panel
page1 = ui.navset_card_underline(
    ui.nav_panel(
        "Demand and Production Capacity",
        ui.output_plot("demand_hist")
    ),
    ui.nav_panel(
        "Customer Sample",
        ui.output_data_frame("sample_bxplt")
    ),
    footer = ui.input_select(
        id = "var",
        label = "Select variable",
        choices = [

        ]
    )
)
