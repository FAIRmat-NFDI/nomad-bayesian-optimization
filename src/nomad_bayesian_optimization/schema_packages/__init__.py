from nomad.config.models.plugins import SchemaPackageEntryPoint


class BayesianOptimizationPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
            m_package,
        )

        return m_package


class SpinCoatingPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.spin_coating import (
            m_package,
        )

        return m_package


bayesian_optimization = BayesianOptimizationPackageEntryPoint(
    name='Bayesian Optimization',
    description='Schema package for Bayesian optimization runs.',
)

spin_coating = SpinCoatingPackageEntryPoint(
    name='Spin Coating',
    description='Schema package for simple spin coating measurements.',
)
