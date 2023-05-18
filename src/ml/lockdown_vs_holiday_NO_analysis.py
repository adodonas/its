import logging
import os
import textwrap

import pandas as pd
from scipy import stats

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
    # Extract NO data for different conditions
    lockdown_NO_data = df[df['lockdown'] == 1]['NO']
    holiday_NO_data = df[df['jewish_holiday'] == 1]['NO']

    # Perform t-test
    t_statistic, p_value = calculate_ttest(lockdown_NO_data, holiday_NO_data)

    # Generate hypothesis text
    hypothesis_text = generate_hypothesis_text(p_value)

    # Prepare data for mean comparison plot and save it
    conditions = ['Lockdown', 'Jewish Holiday']
    averages = [lockdown_NO_data.mean(), holiday_NO_data.mean()]

    log_and_return(f"Mean NO levels during lockdowns and Jewish holidays: {dict(zip(conditions, averages))}")

    # Generate and save markdown file
    markdown_filename = "analysis.md"
    with open(markdown_filename, 'w') as file:
        file.write("# Analysis\n")
        file.write(
            "This report presents a statistical analysis comparing Nitric Oxide (NO) levels during lockdowns "
            "and Jewish holidays.\n")
        file.write("\n")
        file.write("## Hypothesis\n")
        file.write(
            "In scientific research, a hypothesis is a proposed explanation for a phenomenon. For this analysis, "
            "the hypothesis is that the average NO levels are significantly different during lockdowns and Jewish "
            "holidays.\n")
        file.write("\n")
        file.write(
            "This is considered as the alternative hypothesis (H1). On the contrary, the null hypothesis (H0) is "
            "that there is no significant difference in the average NO levels during lockdowns and Jewish holidays. T"
            "he null hypothesis assumes that any kind of difference or importance you see in your data is due to "
            "chance.\n")
        file.write("\n")
        file.write(f"{hypothesis_text}\n")
        file.write("\n")
        file.write("## Mean NO levels\n")
        file.write("The following are the average NO levels during the two different periods:\n")
        file.write(f"During lockdown: {averages[0]}, During Jewish holidays: {averages[1]}\n")
        file.write("\n")
        file.write("## Conclusion\n")
        file.write(
            f"The p-value was {p_value}. This p-value is used to determine the statistical significance of our "
            f"results. The p-value is a measure of the probability that an observed difference could have occurred "
            f"just by random chance.\n")
        file.write("\n")
        file.write(
            "A lower p-value suggests that the observed data are less likely to occur under the null hypothesis. "
            "Therefore, a p-value less than or equal to 0.05 is often used as a cut-off for rejecting the null "
            "hypothesis. In other words, if the p-value is less than or equal to 0.05, we consider the results "
            "statistically significant, and we reject the null hypothesis in favor of the alternative hypothesis.\n")
        file.write("\n")
        file.write(
            "If the p-value is greater than 0.05, we consider the results not statistically significant, and we "
            "do not reject the null hypothesis. This means that the observed difference in NO levels could have "
            "occurred by chance, and there is no evidence to suggest that the NO levels are significantly different "
            "during lockdowns and Jewish holidays.\n")
        file.write("\n")
        file.write("Interpretation of this result is left as an exercise for the reader.\n")

    logging.info(f"Markdown file saved as {markdown_filename}.")
    return markdown_filename


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

    logging.info(f'Summary saved as {summary_filename}. Done.')


perform_statistical_analysis()
