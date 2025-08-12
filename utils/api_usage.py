import requests
import time

# Function to check the task status
def check_task_status(task_id):
    # API endpoint to check task status
    status_url = "http://localhost:8000/task_status/"  # Replace with your API endpoint
    try:
        response = requests.get(f"{status_url}{task_id}/")
        response.raise_for_status()
        return response.json()  # Assuming the response contains a JSON with 'status' and 'result'
    except requests.exceptions.RequestException as e:
        print("Error checking task status:", e)
        return None

# Task polling function
def poll_task_status(task_id):
    while True:
        status = check_task_status(task_id)
        if status is None:
            print("Failed to fetch status. Exiting.")
            break

        # print("Current task status:", status["status"])

        if status["status"] not in ["PENDING", "STARTED"]:
            if status["status"] == "SUCCESS":
                # print("Task completed successfully!")
                # print("Result:", status["result"])
                return status["result"]
            elif status["status"] == "FAILURE":
                print("Task failed.")
                print("Error:", status.get("result", "No error details provided."))
            break

        # Wait for 1 second before the next poll
        time.sleep(1)

def submit_task_and_poll(endpoint_url, payload, headers=None):
    """
    Submits a task to the given API endpoint and polls for the task result.

    Args:
        endpoint_url (str): The API endpoint to submit the task and check status.
        payload (dict): The payload to send in the POST request.
        headers (dict, optional): Additional headers for the POST request.

    Returns:
        dict: The task result if successful or an error message.
    """
    headers = headers or {"Content-Type": "application/json"}

    # Define the patient description
    payload = {
        "query": payload  
    }
    
    try:
        # Submit the task
        response = requests.post(f"http://localhost:8000/{endpoint_url}/", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        task_id = data.get("task_id")
        
        if not task_id:
            return {"error": "No task_id returned from API."}
        
        # print("Task submitted. Task ID:", task_id)

        result = poll_task_status(task_id)
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def get_relevant(patient_description):

    related_hpo_terms = submit_task_and_poll("retrieve_related_hpo_terms", patient_description)

    relevant_submission = {'query': patient_description, 'hpos': related_hpo_terms}
    relevant_hpo_terms = submit_task_and_poll("retrieve_relevant_hpo_terms", relevant_submission)
    return relevant_hpo_terms, related_hpo_terms


