import dataclasses

setup_paths = None


def setup_scanvi(
    model_params_file: str,
    data_path: str,
    config_path: str,
):
    """
    Setup the scanvi pipeline.
    """
    # setup_paths(
    #     model_params_file=model_params_file,
    #     data_path=data_path,
    #     config_path=config_path,
    # )
    pass


def setup_scvi(
    model_params_file: str,
    data_path: str,
    config_path: str,
):
    """
    Setup the scanvi pipeline.
    """
    setup_paths(
        model_params_file=model_params_file,
        data_path=data_path,
        config_path=config_path,
    )


def setup_e2e(
    model_params_file: str,
    data_path: str,
    config_path: str,
):
    """
    Setup the e2e pipeline.
    """
    setup_paths(
        model_params_file=model_params_file,
        data_path=data_path,
        config_path=config_path,
    )


def setup_xgboost(
    model_params_file: str,
    data_path: str,
    config_path: str,
):
    """
    Setup the xgboost pipeline.
    """
    setup_paths(
        model_params_file=model_params_file,
        data_path=data_path,
        config_path=config_path,
    )


def setup_lbl8r(
    model_params_file: str,
    data_path: str,
    config_path: str,
):
    """
    Setup the lbl8r pipeline.
    """
    setup_paths(
        model_params_file=model_params_file,
        data_path=data_path,
        config_path=config_path,
    )
