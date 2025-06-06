import pytest
import asyncio
import json
import time
from main import (
    EventManager,
    EventType,
    RunStartedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    RunFinishedEvent,
    BaseEvent
)

@pytest.fixture
def event_manager():
    return EventManager()

@pytest.fixture
def thread_id():
    return "test_thread_123"

@pytest.fixture
def run_id():
    return "test_run_456"

@pytest.fixture
def message_id():
    return "test_message_789"

def test_event_sequence_generation(event_manager):
    """Test that sequence IDs are generated correctly."""
    # Create a series of events
    events = []
    for _ in range(5):
        event = BaseEvent(
            type=EventType.RUN_STARTED,
            thread_id="test",
            run_id="test"
        )
        # Prepare the event to get a sequence ID
        prepared_event = event_manager.prepare_event(event)
        events.append(prepared_event)
    
    # Verify sequence IDs are sequential
    for i in range(1, len(events)):
        assert events[i].sequence_id > events[i-1].sequence_id, "Sequence IDs should be increasing"

def test_event_correlation(event_manager, thread_id, run_id, message_id):
    """Test event correlation tracking."""
    # Create a run started event
    run_started = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id
    )
    # Prepare the event
    run_started = event_manager.prepare_event(run_started)
    
    # Create a message event with correlation
    message_event = TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        role="assistant",
        correlation_id=run_started.correlation_id
    )
    # Prepare the message event
    message_event = event_manager.prepare_event(message_event)
    
    # Verify correlation
    assert message_event.correlation_id == run_started.correlation_id
    correlated_events = event_manager.get_correlated_events(run_started.correlation_id)
    assert len(correlated_events) == 2
    assert run_started in correlated_events
    assert message_event in correlated_events

def test_event_validation(event_manager, thread_id, run_id, message_id):
    """Test event validation rules."""
    # Test valid event
    valid_event = TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        role="assistant"
    )
    # Prepare and validate the event
    valid_event = event_manager.prepare_event(valid_event)
    is_valid, message = event_manager.validate_event(valid_event)
    assert is_valid, f"Valid event failed validation: {message}"

    # Test invalid event (missing required fields)
    # Create a dict with missing required fields
    invalid_event_dict = {
        "type": EventType.TEXT_MESSAGE_START,
        "thread_id": thread_id,
        "run_id": run_id
    }
    # Create the event using model_construct to bypass validation
    invalid_event = TextMessageStartEvent.model_construct(**invalid_event_dict)
    print(f"\nInvalid event created with model_construct:")
    print(f"Event type: {type(invalid_event)}")
    print(f"Event attributes: {invalid_event.model_dump()}")
    
    # Validate the event
    is_valid, message = event_manager.validate_event(invalid_event)
    print(f"Validation result: is_valid={is_valid}, message={message}")
    
    assert not is_valid, "Invalid event passed validation"
    assert "Invalid event type requirements" in message, f"Unexpected validation message: {message}"

    # Test invalid event (null required fields)
    null_event_dict = {
        "type": EventType.TEXT_MESSAGE_START,
        "thread_id": thread_id,
        "run_id": run_id,
        "message_id": None,
        "role": None
    }
    null_event = TextMessageStartEvent.model_construct(**null_event_dict)
    is_valid, message = event_manager.validate_event(null_event)
    assert not is_valid, "Event with null required fields passed validation"
    assert "Invalid event type requirements" in message

    # Test invalid event (wrong type)
    wrong_type_dict = {
        "type": EventType.TEXT_MESSAGE_END,  # Wrong type for TextMessageStartEvent
        "thread_id": thread_id,
        "run_id": run_id,
        "message_id": message_id,
        "role": "assistant"
    }
    wrong_type_event = TextMessageStartEvent.model_construct(**wrong_type_dict)
    is_valid, message = event_manager.validate_event(wrong_type_event)
    assert not is_valid, "Event with wrong type passed validation"
    assert "Invalid event type requirements" in message

    # Test invalid event (invalid correlation)
    # First create an event with a specific correlation ID
    first_correlation_event = TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        thread_id=thread_id,
        run_id=run_id,
        message_id="first_msg",
        role="assistant"
    )
    # Prepare it to get a correlation ID and sequence ID
    first_correlation_event = event_manager.prepare_event(first_correlation_event)
    print(f"\nFirst correlation event: correlation_id={first_correlation_event.correlation_id}, sequence_id={first_correlation_event.sequence_id}")
    
    # Now create an invalid event that tries to use the same correlation ID but with a lower sequence ID
    invalid_correlation_dict = {
        "type": EventType.TEXT_MESSAGE_START,
        "thread_id": thread_id,
        "run_id": run_id,
        "message_id": message_id,
        "role": "assistant",
        "correlation_id": first_correlation_event.correlation_id,  # Use existing correlation ID
        "sequence_id": first_correlation_event.sequence_id - 1  # Lower sequence ID - this should fail correlation validation
    }
    print(f"\nCreating invalid correlation event with dict: {invalid_correlation_dict}")
    invalid_correlation_event = TextMessageStartEvent.model_construct(**invalid_correlation_dict)
    print(f"Created event attributes: {invalid_correlation_event.model_dump()}")
    print(f"Event correlation_id: {invalid_correlation_event.correlation_id}")
    print(f"Event sequence_id: {invalid_correlation_event.sequence_id}")
    
    is_valid, message = event_manager.validate_event(invalid_correlation_event)
    print(f"Validation result: is_valid={is_valid}, message={message}")
    assert not is_valid, "Event with invalid correlation passed validation"
    assert "Invalid event correlation" in message

def test_event_preparation(event_manager, thread_id, run_id):
    """Test event preparation and serialization."""
    # Create and prepare an event
    event = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id,
        agent_metadata={"name": "test_agent"}
    )
    # Prepare the event
    prepared_event = event_manager.prepare_event(event)
    
    # Verify preparation
    assert prepared_event.sequence_id is not None
    assert prepared_event.correlation_id is not None
    
    # Test serialization
    serialized = event_manager.serialize_event(prepared_event)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["type"] == EventType.RUN_STARTED
    assert deserialized["thread_id"] == thread_id
    assert deserialized["run_id"] == run_id

def test_event_history(event_manager, thread_id, run_id):
    """Test event history management."""
    # Create and send multiple events
    events = []
    for i in range(3):
        event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            thread_id=thread_id,
            run_id=run_id,
            message_id=f"msg_{i}",
            role="assistant",
            delta=f"Test message {i}"
        )
        # Prepare the event
        prepared_event = event_manager.prepare_event(event)
        events.append(prepared_event)
    
    # Verify history
    history = event_manager.get_event_history(thread_id=thread_id)
    assert len(history) == 3
    for event in events:
        assert event in history
    
    # Test clearing history
    event_manager.clear_history(thread_id=thread_id)
    assert len(event_manager.get_event_history(thread_id=thread_id)) == 0

def test_complete_event_flow(event_manager, thread_id, run_id, message_id):
    """Test a complete event flow from start to finish."""
    # Start run
    run_started = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id,
        agent_metadata={"name": "test_agent"}
    )
    # Prepare the event
    run_started = event_manager.prepare_event(run_started)
    
    # Start message
    message_start = TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        role="assistant",
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    message_start = event_manager.prepare_event(message_start)
    
    # Send content
    content = TextMessageContentEvent(
        type=EventType.TEXT_MESSAGE_CONTENT,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        role="assistant",
        delta="Test message",
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    content = event_manager.prepare_event(content)
    
    # End message
    message_end = TextMessageEndEvent(
        type=EventType.TEXT_MESSAGE_END,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    message_end = event_manager.prepare_event(message_end)
    
    # End run
    run_finished = RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=thread_id,
        run_id=run_id,
        completion_status="success",
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    run_finished = event_manager.prepare_event(run_finished)
    
    # Verify event sequence
    history = event_manager.get_event_history(thread_id=thread_id)
    assert len(history) == 5
    assert history[0] == run_started
    assert history[1] == message_start
    assert history[2] == content
    assert history[3] == message_end
    assert history[4] == run_finished
    
    # Verify correlation
    correlated_events = event_manager.get_correlated_events(run_started.correlation_id)
    assert len(correlated_events) == 5
    assert all(event.correlation_id == run_started.correlation_id for event in correlated_events)

def test_error_handling(event_manager, thread_id, run_id, message_id):
    """Test error handling in event flow."""
    # Start run
    run_started = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id
    )
    # Prepare the event
    run_started = event_manager.prepare_event(run_started)
    
    # Create an error event
    error_event = TextMessageContentEvent(
        type=EventType.TEXT_MESSAGE_CONTENT,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        role="assistant",
        delta="Error occurred",
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    error_event = event_manager.prepare_event(error_event)
    
    # End message with error
    message_end = TextMessageEndEvent(
        type=EventType.TEXT_MESSAGE_END,
        thread_id=thread_id,
        run_id=run_id,
        message_id=message_id,
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    message_end = event_manager.prepare_event(message_end)
    
    # End run with error
    run_finished = RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=thread_id,
        run_id=run_id,
        completion_status="error",
        correlation_id=run_started.correlation_id
    )
    # Prepare the event
    run_finished = event_manager.prepare_event(run_finished)
    
    # Verify error flow
    history = event_manager.get_event_history(thread_id=thread_id)
    assert len(history) == 4
    assert history[0] == run_started
    assert history[1] == error_event
    assert history[2] == message_end
    assert history[3] == run_finished
    assert run_finished.completion_status == "error"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 