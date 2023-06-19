import logging
import os
import textwrap

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

# Set logging level
logging.basicConfig(level=logging.INFO)


def log_and_return(message):
    logging.info(message)
    return message


def calculate_ttest(lockdown_data, jewish_holiday_data):
    t_stat, p_value = stats.ttest_ind(lockdown_data, jewish_holiday_data)
    log_and_return(f"t-statistic: {t_stat}, p-value: {p_value}")
    return t_stat, p_value


def generate_hypothesis_text(p_value, p_value_threshold=0.05):
    base_text = "The p-value is {} 0.05, which suggests that the NO levels during lockdowns and Jewish holidays " \
                "are {} significantly different."
    result_text = base_text.format("less than or equal to" if p_value <= p_value_threshold else "greater than",
                                   "" if p_value <= p_value_threshold else "not")

    return log_and_return('\n'.join(textwrap.wrap(result_text, width=150)))


def calculate_and_save_summary(df):
    # Exclude Jewish holidays during lockdown and vice versa
    lockdown_NO_data = df[(df['lockdown'] == 1) & (df['jewish_holiday'] == 0)]['NO']
    holiday_NO_data = df[(df['jewish_holiday'] == 1) & (df['lockdown'] == 0)]['NO']

    # Perform t-test
    t_statistic, p_value = calculate_ttest(lockdown_NO_data, holiday_NO_data)

    # Generate hypothesis text
    hypothesis_text = generate_hypothesis_text(p_value)

    # Prepare data for mean comparison plot and save it
    conditions = ['Lockdown', 'Jewish Holiday']
    averages = [lockdown_NO_data.mean(), holiday_NO_data.mean()]

    log_and_return(f"Mean NO during: {dict(zip(conditions, averages))}")

    # Generate and save markdown file
    markdown_filename = "analysis.md"
    with open(markdown_filename, 'w') as file:
        file.write("# Analysis\n")
        file.write("This report evaluates NO levels during lockdowns and Jewish holidays, separately.\n")
        file.write("\n")
        file.write("## Hypothesis\n")
        file.write(
            "The hypothesis for this analysis is that the average NO levels differ between lockdowns and Jewish "
            "holidays.\n")
        file.write(f"{hypothesis_text}\n")
        file.write("\n")
        file.write("## Mean NO levels\n")
        file.write("The average NO levels during each period are:\n")
        file.write(f"Lockdown: {averages[0]}, Jewish holidays: {averages[1]}\n")
        file.write("\n")
        file.write("## Conclusion\n")
        file.write(
            f"The p-value was {p_value}, measuring the likelihood that observed differences occurred by chance.\n")
        file.write("\n")
        file.write(
            "A p-value ≤ 0.05 is often used as a threshold for statistical significance, rejecting the null "
            "hypothesis in favor of the alternative one.\n")
        file.write("\n")
        file.write(
            "A p-value > 0.05 indicates the observed difference in NO levels could be due to chance, suggesting "
            "no significant difference between the periods.\n")

    logging.info(f"Markdown file saved as {markdown_filename}.")
    return markdown_filename


def plot_NO_levels_all(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract relevant data
    lockdown = df[df['lockdown'] == 1]
    holiday = df[df['jewish_holiday'] == 1]
    regular_days = df[(df['lockdown'] == 0) & (df['jewish_holiday'] == 0)]

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({
        'NO': pd.concat([lockdown['NO'], holiday['NO'], regular_days['NO']], ignore_index=True),
        'Period': pd.concat([
            pd.Series(['Lockdown'] * len(lockdown)),
            pd.Series(['Jewish Holiday'] * len(holiday)),
            pd.Series(['Regular Days'] * len(regular_days))
        ], ignore_index=True)
    })

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='NO', data=plot_df)
    plt.title('NO levels for different periods')
    plt.ylabel('NO level')
    plt.xlabel('Period')

    # Save the plot
    plt.savefig('NO_levels.png')
    logging.info("Plot saved as 'NO_levels.png'.")


def preprocess_dataframe(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def create_date_ranges(*dates):
    date_ranges = pd.date_range(start=dates[0], end=dates[1])
    for i in range(2, len(dates), 2):
        date_ranges = date_ranges.append(pd.date_range(start=dates[i], end=dates[i + 1]))
    return date_ranges.date


def filter_data(df, dates):
    return df[df['date'].isin(dates)]


def create_plot_df(df, lockdown, yom_kippur, other_holidays):
    return pd.concat([
        pd.DataFrame({'NO': lockdown['NO'], 'Period': 'Lockdowns'}),
        pd.DataFrame({'NO': yom_kippur['NO'], 'Period': 'Yom Kippur'}),
        pd.DataFrame({'NO': other_holidays['NO'], 'Period': 'Other Holidays'})
    ], ignore_index=True)


def plot_NO_levels(df):
    df = preprocess_dataframe(df)

    lockdown_dates = create_date_ranges("2020-03-17", "2020-04-19", "2020-09-11", "2020-10-13", "2020-12-27",
                                        "2021-02-07")
    yom_kippur_dates = create_date_ranges("2018-09-18", "2018-09-19", "2019-10-08", "2019-10-09", "2020-09-27",
                                          "2020-09-28", "2021-09-15", "2021-09-16", "2022-10-04", "2022-10-05")
    additional_holidays = create_date_ranges("2018-05-19", "2018-05-20", "2019-06-08", "2019-06-10", "2020-05-28",
                                             "2020-05-30", "2021-05-16", "2021-05-18", "2022-06-04", "2022-06-06",
                                             "2018-03-30", "2018-04-07", "2019-04-19", "2019-04-27", "2020-04-08",
                                             "2020-04-16", "2021-03-27", "2021-04-04", "2022-04-15", "2022-04-23",
                                             "2018-09-09", "2018-09-11", "2019-09-29", "2019-10-01", "2020-09-18",
                                             "2020-09-20", "2021-09-06", "2021-09-08", "2022-09-25", "2022-09-27",
                                             "2018-09-23", "2018-10-02", "2019-10-13", "2019-10-22", "2020-10-02",
                                             "2020-10-11", "2021-09-20", "2021-09-29", "2022-10-09", "2022-10-18")

    lockdown = filter_data(df, lockdown_dates)
    yom_kippur = filter_data(df, yom_kippur_dates)
    other_holidays = filter_data(df, additional_holidays)

    plot_df = create_plot_df(df, lockdown, yom_kippur, other_holidays)

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='NO', data=plot_df)
    plt.title('NO levels for Lockdowns, Yom Kippur and Other Holidays')
    plt.ylabel('NO level')
    plt.xlabel('Period')

    # Save the plot
    plt.savefig('NO_levels_specific_periods.png')
    logging.info("Plot saved as 'NO_levels_specific_periods.png'.")


def plot_NO_levels_stem(df):
    # Convert 'timestamp' column to datetime and set it as index
    df.set_index('timestamp', inplace=True)

    # Define date ranges for the events
    lockdown_dates = create_date_ranges("2020-03-17", "2020-04-19", "2020-09-11", "2020-10-13", "2020-12-27",
                                        "2021-02-07")
    yom_kippur_dates = create_date_ranges("2018-09-18", "2018-09-19", "2019-10-08", "2019-10-09", "2020-09-27",
                                          "2020-09-28", "2021-09-15", "2021-09-16", "2022-10-04", "2022-10-05")
    additional_holidays = create_date_ranges("2018-05-19", "2018-05-20", "2019-06-08", "2019-06-10", "2020-05-28",
                                             "2020-05-30", "2021-05-16", "2021-05-18", "2022-06-04", "2022-06-06",
                                             "2018-03-30", "2018-04-07", "2019-04-19", "2019-04-27", "2020-04-08",
                                             "2020-04-16", "2021-03-27", "2021-04-04", "2022-04-15", "2022-04-23",
                                             "2018-09-09", "2018-09-11", "2019-09-29", "2019-10-01", "2020-09-18",
                                             "2020-09-20", "2021-09-06", "2021-09-08", "2022-09-25", "2022-09-27",
                                             "2018-09-23", "2018-10-02", "2019-10-13", "2019-10-22", "2020-10-02",
                                             "2020-10-11", "2021-09-20", "2021-09-29", "2022-10-09", "2022-10-18")

    # Filter the data according to the events
    lockdown_data = filter_data(df, lockdown_dates)
    yom_kippur_data = filter_data(df, yom_kippur_dates)
    other_holidays_data = filter_data(df, additional_holidays)

    # Create stem plots
    plt.figure(figsize=(16, 8))

    # Stem plot for lockdown
    markerline, stemlines, _ = plt.stem(lockdown_data.index, lockdown_data['NO'], linefmt='-', basefmt=" ",
                                        label='Lockdowns')
    plt.setp(stemlines, 'linewidth', 1)

    # Stem plot for Yom Kippur
    markerline, stemlines, _ = plt.stem(yom_kippur_data.index, yom_kippur_data['NO'], linefmt='-', basefmt=" ",
                                        label='Yom Kippur')
    plt.setp(stemlines, 'linewidth', 1)

    # Stem plot for other holidays
    markerline, stemlines, _ = plt.stem(other_holidays_data.index, other_holidays_data['NO'], linefmt='-', basefmt=" ",
                                        label='Other Holidays')
    plt.setp(stemlines, 'linewidth', 1)

    # Setting the title and labels
    plt.title('NO levels for Lockdowns, Yom Kippur and Other Holidays')
    plt.ylabel('NO level')
    plt.xlabel('Period')
    plt.legend()

    # Save the plot
    plt.savefig('NO_levels_stem_plot.png')
    logging.info("Plot saved as 'NO_levels_stem_plot.png'.")


def plot_NO_levels_stem(df):
    # Convert 'timestamp' column to datetime and set it as index
    df.set_index('timestamp', inplace=True)

    lockdown_data, passover_data, rosh_hashana_data, shavuot_data, sukkot_data, yom_kippur_data = get_data(df)

    # Create stem plots
    plt.figure(figsize=(16, 8))

    # Stem plot for each event
    plot_stem(lockdown_data, 'NO', 'Lockdowns')
    plot_stem(yom_kippur_data, 'NO', 'Yom Kippur')
    plot_stem(shavuot_data, 'NO', 'Shavuot')
    plot_stem(passover_data, 'NO', 'Passover')
    plot_stem(rosh_hashana_data, 'NO', 'Rosh Hashana')
    plot_stem(sukkot_data, 'NO', 'Sukkot')

    # Setting the title and labels
    plt.title('NO levels for Different Events')
    plt.ylabel('NO level')
    plt.xlabel('Period')
    plt.legend()

    # Save the plot
    plt.savefig('NO_levels_stem_plot.png')
    logging.info("Plot saved as 'NO_levels_stem_plot.png'.")


def get_data(df):
    # Define date ranges for the events
    lockdown_dates = create_lockdown_dates()
    yom_kippur_dates = create_yom_kippur_dates()
    shavuot_dates = create_shavuot_dates()
    passover_dates = create_passover_dates()
    rosh_hashana_dates = create_rosh_hashana_dates()
    sukkot_dates = create_sukkot_dates()
    # Filter the data according to the events
    lockdown_data = filter_data(df, lockdown_dates)
    yom_kippur_data = filter_data(df, yom_kippur_dates)
    shavuot_data = filter_data(df, shavuot_dates)
    passover_data = filter_data(df, passover_dates)
    rosh_hashana_data = filter_data(df, rosh_hashana_dates)
    sukkot_data = filter_data(df, sukkot_dates)
    return lockdown_data, passover_data, rosh_hashana_data, shavuot_data, sukkot_data, yom_kippur_data


def plot_stem(data, column, label):
    markerline, stemlines, _ = plt.stem(data.index, data[column], linefmt='-', basefmt=" ", label=label)
    plt.setp(stemlines, 'linewidth', 1)


def create_lockdown_dates():
    return create_date_ranges("2020-03-17", "2020-04-19", "2020-09-11", "2020-10-13", "2020-12-27",
                              "2021-02-07")


def create_yom_kippur_dates():
    return create_date_ranges("2018-09-18", "2018-09-19", "2019-10-08", "2019-10-09", "2020-09-27",
                              "2020-09-28", "2021-09-15", "2021-09-16", "2022-10-04", "2022-10-05")


def create_shavuot_dates():
    return create_date_ranges("2018-05-19", "2018-05-20", "2019-06-08", "2019-06-10", "2020-05-28",
                              "2020-05-30", "2021-05-16", "2021-05-18", "2022-06-04", "2022-06-06")


def create_passover_dates():
    return create_date_ranges("2018-03-30", "2018-04-07", "2019-04-19", "2019-04-27", "2020-04-08",
                              "2020-04-16", "2021-03-27", "2021-04-04", "2022-04-15", "2022-04-23")


def create_rosh_hashana_dates():
    return create_date_ranges("2018-09-09", "2018-09-11", "2019-09-29", "2019-10-01",
                              "2020-09-18", "2020-09-20", "2021-09-06", "2021-09-08", "2022-09-25",
                              "2022-09-27")


def create_sukkot_dates():
    return create_date_ranges("2018-09-23", "2018-10-02", "2019-10-13", "2019-10-22", "2020-10-02",
                              "2020-10-11", "2021-09-20", "2021-09-29", "2022-10-09", "2022-10-18")


def create_stem_plots(lockdown_data, yom_kippur_data, other_holidays_data):
    plt.figure(figsize=(16, 8))

    # Stem plot for lockdown
    create_individual_stem_plot(lockdown_data, 'NO', '-', ' ', 'Lockdowns')

    # Stem plot for Yom Kippur
    create_individual_stem_plot(yom_kippur_data, 'NO', '-', ' ', 'Yom Kippur')

    # Stem plot for other holidays
    create_individual_stem_plot(other_holidays_data, 'NO', '-', ' ', 'Other Holidays')

    # Setting the title and labels
    plt.title('NO levels for Lockdowns, Yom Kippur and Other Holidays')
    plt.ylabel('NO level')
    plt.xlabel('Period')
    plt.legend()

    # Save the plot
    plt.savefig('NO_levels_stem_plot.png')
    logging.info("Plot saved as 'NO_levels_stem_plot.png'.")


def create_individual_stem_plot(data, column, linefmt, basefmt, label):
    markerline, stemlines, _ = plt.stem(data.index, data[column], linefmt=linefmt, basefmt=basefmt, label=label)
    plt.setp(stemlines, 'linewidth', 1)


def calculate_regular_NO_level(df):
    # Convert 'timestamp' column to datetime if it's not
    if df.index.dtype != 'datetime64[ns]':
        df.set_index(pd.to_datetime(df['timestamp']), inplace=True)

    # Adjust the data for Israeli working days (6 is Sunday and 3 is Thursday in dt.weekday)
    working_days_data = df[(df.index.weekday < 4) | (df.index.weekday == 6)]

    # Calculate the average NO level for working days
    regular_NO_level = working_days_data['NO'].mean()
    logging.info(f'Average NO level for working days: {regular_NO_level}')
    return regular_NO_level


def perform_statistical_analysis():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set the directory of the file (make sure to change to your specific directory)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, '../feature_engineering/results/final_normal_merged.xlsx')

    try:
        # Load the data from the Excel file
        logging.info('Loading data...')
        df = pd.read_excel(file_path)
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        raise

    # Check if necessary columns exist
    assert 'timestamp' in df.columns, "'timestamp' column is missing in the dataframe"

    # Convert the timestamps to datetime objects
    logging.info('Converting timestamps to datetime objects...')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Perform the t-test and create a summary
    logging.info('Performing analysis...')
    summary_filename = calculate_and_save_summary(df)

    # Create and save the plot
    logging.info('Creating plot...')
    plot_NO_levels(df)
    plot_NO_levels_stem(df)
    calculate_regular_NO_level(df)
    logging.info(f'Summary saved as {summary_filename}. Done.')


perform_statistical_analysis()
