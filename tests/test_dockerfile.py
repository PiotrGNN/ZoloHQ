def test_dockerfile_paths():
    with open('Dockerfile') as f:
        content = f.read()
    assert 'ZoL0-master' not in content
