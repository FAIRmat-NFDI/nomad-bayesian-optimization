from nomad.config.models.ui import (
    App,
    Axis,
    Column,
    Menu,
    MenuItemHistogram,
    MenuItemTerms,
    SearchQuantities,
)

schema_name = 'nomad_bayesian_optimization.schema_packages.bayesian_optimization.BayesianOptimization'  # noqa: E501
app = App(
    label='Bayesian Optimizations Tasks',
    path='bayesian-optimization-tasks',
    category='Bayesian Optimization',
    search_quantities=SearchQuantities(include=[f'*#{schema_name}']),
    columns=[
        Column(search_quantity='entry_create_time', selected=True),
        Column(search_quantity=f'data.status#{schema_name}', selected=True),
        Column(
            search_quantity=f'data.objective.target.name#{schema_name}', selected=True
        ),
        Column(
            search_quantity=f'data.parameters[*].name#{schema_name}',
            selected=True,
        ),
    ],
    menu=Menu(
        items=[
            MenuItemTerms(
                search_quantity=f'data.status#{schema_name}',
                show_input=False,
            ),
            Menu(
                title='Objective',
                items=[
                    MenuItemTerms(
                        search_quantity=f'data.objective.target.type#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.objective.target.name#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.objective.target.mode#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.objective.target.transformation#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemHistogram(
                        x=Axis(
                            search_quantity=f'data.objective.target.bounds.lower#{schema_name}'
                        )
                    ),
                    MenuItemHistogram(
                        x=Axis(
                            search_quantity=f'data.objective.target.bounds.upper#{schema_name}'
                        )
                    ),
                ],
            ),
            Menu(
                size='xs',
                title='Parameters',
                items=[
                    MenuItemTerms(
                        search_quantity=f'data.parameters.name#{schema_name}',
                    ),
                ],
            ),
            Menu(
                title='Recommender',
                items=[
                    MenuItemTerms(
                        search_quantity=f'data.recommender.type#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.recommender.surrogate_model.type#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.recommender.acquisition_function.type#{schema_name}',
                        show_input=False,
                        options=1,
                    ),
                    MenuItemTerms(
                        search_quantity=f'data.recommender.hybrid_sampler#{schema_name}',
                        show_input=False,
                    ),
                    MenuItemHistogram(
                        x=Axis(
                            search_quantity=f'data.recommender.sampling_percentage#{schema_name}'
                        )
                    ),
                ],
            ),
        ]
    ),
    filters_locked={'section_defs.definition_qualified_name': [schema_name]},
)
