from nomad.config.models.plugins import SchemaPackageEntryPoint


class BayesianOptimizationPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
            m_package,
        )

        return m_package


class ExperimentsPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.experiments import m_package

        return m_package


experiments = ExperimentsPackageEntryPoint(
    name='Experiments',
    description='Dummy schema package for experiments.',
)

bayesian_optimization = BayesianOptimizationPackageEntryPoint(
    name='Bayesian Optimization',
    description='Schema package for Bayesian optimization runs.',
)
