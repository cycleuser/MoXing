from moxing.binaries import (
    BinaryManager,
    PlatformDetector,
    get_binary_manager,
    list_available_backends,
)


def test_platform_detector_get_platform_name():
    name = PlatformDetector.get_platform_name()
    assert isinstance(name, str)
    assert len(name) > 0
    assert "-" in name


def test_platform_detector_get_os():
    os_name = PlatformDetector.get_os()
    assert os_name in ("darwin", "linux", "windows")


def test_platform_detector_get_arch():
    arch = PlatformDetector.get_arch()
    assert isinstance(arch, str)
    assert len(arch) > 0
    assert arch in ("arm64", "x64")


def test_binary_manager_constructor():
    manager = BinaryManager()
    assert isinstance(manager, BinaryManager)
    assert manager._requested_backend == "auto"


def test_get_binary_manager():
    manager = get_binary_manager()
    assert isinstance(manager, BinaryManager)


def test_list_available_backends():
    result = list_available_backends()
    assert isinstance(result, dict)
    for backend in result:
        assert isinstance(backend, str)
        assert isinstance(result[backend], bool)
