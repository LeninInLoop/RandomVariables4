# Standard library imports
from collections import Counter
from typing import Dict, List, Tuple, Union

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import chi2

# Counting methods
METHOD_MANUAL = 0
METHOD_COUNTER_CLASS = 1

def initialize_value_count_dict(x_range:Tuple[Union[int],Union[int]] = (0,1)) -> Dict[float, int]:
    """
    Initialize a dictionary for counting occurrences of each bin value.

    Returns:
    Dict[float, int]: A dictionary with bin values as keys and 0 as initial counts.
    """
    values = np.linspace(x_range[0],x_range[1],1000).tolist()
    return {round_to_decimals(value, 3):0 for value in values}

def round_to_decimals(value: float, decimal_places: int = 0) -> float:
    """
    Custom round function to round a number to the specified number of decimal places.

    This function handles both positive and negative numbers correctly, ensuring
    that numbers are rounded to the nearest value, providing consistent results for all inputs.

    Parameters:
    value (float): The number to be rounded.
    decimal_places (int): The number of decimal places to round to (default is 0).

    Returns:
    float: The rounded number.
    """
    multiplier = 10 ** decimal_places
    if value >= 0:
        return (value * multiplier + 0.5) // 1 / multiplier
    else:
        return -((-value * multiplier + 0.5) // 1) / multiplier

def count_values_by_method(
        random_floats: List[float],
        method: int,
        x_range:Tuple[Union[int],Union[int]] = (0,1)
) -> Dict[float, int]:
    """
    Count occurrences of rounded random floats in the specified value bins using a specified method.

    Parameters:
    value_bins (List[float]): A list of bin values to count occurrences against.
    random_floats (List[float]): A list of random float values to be counted.
    method (int): The counting method to use.
                  METHOD_MANUAL (1) uses a manual O(n^2) algorithm.
                  METHOD_COUNTER_CLASS (2) uses Python's Counter class for an O(n log n) approach.

    Returns:
    Dict[float, int]: A dictionary where keys are bin values and values are the counts of occurrences.
    """
    value_count_dict = initialize_value_count_dict(x_range)
    rounded_random_floats = [round_to_decimals(value, 3) for value in random_floats]
    if method == METHOD_MANUAL:  # O(n^2) Algorithm
        for bin_value in value_count_dict.keys():
            for random_float in rounded_random_floats:
                if random_float == bin_value:
                    value_count_dict[bin_value] += 1
    elif method == METHOD_COUNTER_CLASS:  # O(n log n) Algorithm
        random_counts = Counter(rounded_random_floats)
        for number, count in random_counts.items():
            if number in value_count_dict:
                value_count_dict[number] += count
    return value_count_dict

def calculate_cdf_from_pdf(pdf_values: List[float], x_values: List[float]) -> np.ndarray:
    """
    Calculates the cumulative distribution function (CDF) from the probability density function (PDF).

    Parameters:
    pdf_values (List[float]): The values representing the probability density function (PDF).
    x_values (List[float]): The corresponding x values.

    Returns:
    List[float]: The calculated CDF values.
    """
    # Use cumulative trapezoidal integration to calculate the CDF
    return integrate.cumulative_trapezoid(pdf_values, x_values, dx=0.001, initial=0)


def save_pdf_and_cdf_plot_from_pdf(
        value_counts: Dict[float, int],
        display: bool,
        file_name: str = "pdf_and_cdf_plot.png",
):
    """
    Saves a plot with both the Probability Density Function (PDF) in two formats:
    a histogram-based PDF and a line plot, as well as the Cumulative Distribution Function (CDF).

    Parameters:
    value_counts (Dict[float, int]): A dictionary where keys are x-values and values are frequencies (PDF).
    display (bool): If True, display the plot interactively; if False, save the plot as an image.
    filename (str): The name of the file to save the plot as (default is 'pdf_and_cdf_plot.png').
    show_histogram (bool, optional): If True, include a histogram representation of the PDF. Default is True.
    x_range (Tuple[int, int], optional): The range of x-values to use. If None, uses the keys from value_counts. Default is None.

    The function performs the following:
    - If show_histogram is True, plots a histogram to represent the PDF (using the frequency from value_counts).
    - Plots the PDF as a smooth line plot.
    - Computes and plots the CDF based on the PDF data.
    - Adjusts the plot layout based on whether the histogram is shown or not.
    - If x_range is provided, uses it to create value bins; otherwise, uses the keys from value_counts.
    - Displays the plot if `display` is True, or saves it to a file if `display` is False.
    """

    x_values = list(value_counts.keys())
    y_values = [count / 1000 for count in value_counts.values()]

    plt.figure(figsize=(12, 10))

    fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.4)  # Increase space between subplots

    # Plot PDF without Histogram
    ax_pdf.plot(x_values[1:-1], y_values[1:-1], color='blue')
    ax_pdf.fill_between(x_values[1:-1], y_values[1:-1], alpha=0.3)
    ax_pdf.set_xlabel('X Values\n\n')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.set_title('PDF: Probability Density Function')

    # Calculate CDF
    cdf_values = calculate_cdf_from_pdf(y_values, x_values)

    ax_cdf.plot(x_values, cdf_values, color='red')
    ax_cdf.set_xlabel('X Values')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('CDF: Cumulative Distribution Function')

    # Add overall title
    fig.suptitle('Probability Density Function and Cumulative Distribution Function', fontsize=16)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(file_name)
    if display:
        plt.show()
    else:
        plt.close()

def generate_random_variable(mean: float, std: float) -> List[float]:
    return np.array(std) * np.random.randn( 10 ** 6) + np.array(mean) .tolist()

def generate_chi_two_plot(
        degree_of_freedom: int,
        file_name: str = "generated_chi_square.png",
        display: bool = False
):
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 17, 1000)
    chi2_pdf = chi2.pdf(x, df=degree_of_freedom)

    plt.plot(x, chi2_pdf, label='df=' + str(degree_of_freedom), color='blue', linewidth=2)

    plt.title('Chi-square Distribution (Degree of Freedom=' + f"{degree_of_freedom}", fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('Probability Density', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(file_name)
    if display:
        mean, var, skew, kurt = chi2.stats(degree_of_freedom, moments='mvsk')
        print(50 * "-")
        print(f"For chi-square distribution with {degree_of_freedom} degrees of freedom:")
        print(f"\nMean (μ)     = {mean:.4f}")
        print("              • First moment")
        print("              • Average/expected value")
        print("              • For χ²: equals degrees of freedom")

        print(f"\nVariance (σ²) = {var:.4f}")
        print("              • Second moment")
        print("              • Spread of the distribution")
        print("              • For χ²: equals 2 * degrees of freedom")

        print(f"\nSkewness     = {skew:.4f}")
        print("              • Third moment")
        print("              • Measure of asymmetry")
        print("              • For χ²: equals √(8/degree of freedom)")
        print("              • Positive value means right-skewed")

        print(f"\nKurtosis     = {kurt:.4f}")
        print("              • Fourth moment")
        print("              • Measure of heaviness of tails")
        print("              • For χ²: equals 12/degree of freedom")
        print("              • Higher value means heavier tails")
        plt.show()
    else:
        plt.close()

def generate_chi2_square_variables(degree_of_freedom: int) -> List[float]:
    x_vector = np.zeros((degree_of_freedom, 10 ** 6))
    for row in range(0, x_vector.shape[0]):
        x_vector[int(row)] = generate_random_variable(
            mean=0.0,
            std=1.0
        )
    random_variable = np.sum(x_vector ** 2, axis=0)
    return random_variable.tolist()


def main():
    random_variable_z = generate_chi2_square_variables(degree_of_freedom=4)

    counted_values = count_values_by_method(
        random_variable_z,
        method=METHOD_COUNTER_CLASS,
        x_range=(-2, 16)
    )

    save_pdf_and_cdf_plot_from_pdf(
        counted_values,
        display=True,
        file_name="chi_square_df_4.png",
    )

    generate_chi_two_plot(
        degree_of_freedom=4,
        display=True,
        file_name="generated_chi_square_df_4.png",
    )


if __name__ == "__main__" :
    main()