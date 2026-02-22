def test_addition():
    """Test that Python can add numbers."""
    assert 1 + 1 == 2


def test_list_length():
    """Test that Python can count list items."""
    my_list = [1, 2, 3]
    assert len(my_list) == 3


def test_string_concatenation():
    """Test that Python can concatenate strings."""
    result = "Hello" + " " + "World"
    assert result == "Hello World"
