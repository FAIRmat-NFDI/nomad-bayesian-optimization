from nomad.config.models.plugins import SchemaPackageEntryPoint


class BayesianOptimizationPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
            m_package,
        )

        return m_package


bayesian_optimization = BayesianOptimizationPackageEntryPoint(
    name='Bayesian Optimization',
    description='Schema package for Bayesian optimization runs.',
)
