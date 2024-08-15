from nomad.config.models.plugins import AppEntryPoint
from nomad.config.models.ui import App, Column, Columns, FilterMenu, FilterMenus

schema_name = 'nomad_bayesian_optimization.schema_packages.bayesian_optimization.BayesianOptimization'
bayesian_optimization_tasks = AppEntryPoint(
    name='Bayesian Optimizations Tasks',
    description='App for Bayesian Optimization Tasks',
    app=App(
        label='Bayesian Optimizations Tasks',
        path='bayesian-optimization-tasks',
        category='Bayesian Optimization',
        columns=Columns(
            selected=[
                'entry_create_time',
                f'data.status#{schema_name}',
                f'data.objective.target#{schema_name}',
            ],
            options={
                'entry_create_time': Column(),
                f'data.status#{schema_name}': Column(),
                f'data.objective.target#{schema_name}': Column(),
            },
        ),
        filter_menus=FilterMenus(
            options={
                'material': FilterMenu(label='Material'),
            }
        ),
        filters_locked={'section_defs.definition_qualified_name:all': [schema_name]},
    ),
)
