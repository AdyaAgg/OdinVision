def save_feedback(feedback, path="feedback.csv"):
    """
    Appends user feedback to a CSV file.

    Args:
        feedback (str): The feedback text submitted by the user.
        path (str): Path to the CSV file where feedback will be stored.

    Returns:
        str: Confirmation message upon successful write.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{feedback}\n")
    return "Feedback submitted successfully."
