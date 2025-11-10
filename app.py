# Beginning of app.py"

# Import necessary libraries and modules
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
demand = "./data/demand.csv"
production_capacity = "./data/production_capacity.csv"
avg_demand = 183.92


# The contents of the first 'page' is a navset with one panel
page1 = ui.navset_card_underline(
    ui.nav_panel(
        "Demand and Production Capacity",
        ui.output_plot("hist")
    ),
    # ui.nav_panel(
    #     "Customer Sample",
    #     ui.output_data_frame("sample_bxplt")
    # ),
    footer = ui.input_select(
        id = "var",
        label = "Select variable",
        choices = [
            "demand",
            "production_capacity"
        ]
    ),
    title = 'EDA',
)

app_ui = ui.page_navbar(
    ui.nav_spacer(), # Pushing the navbar items to the right
    ui.nav_panel(
        "Page 1",
        page1
    ),
    ui.nav_panel(
        "Page 2",
        "This is a placeholder and second 'page'."
    ),
    title="Dashboard prototype",
)

# Define the server function
def server(
        input,
        output,
        session
):
    @render.plot
    def hist():
        # Plotting Logic
        h = ggplot(
            input.var()
        )
        + aes(
            x=input.var()
        )
        + geom_histogram(
            bins=10,
            fill='#5cb5d3',
            color='#040e05'
        )
        + theme(
            axis_text_x = element_text(rotation=90,hjust=1)
        )
        return h
    
app = App(app_ui, server)

# End of app.py"