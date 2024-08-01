"""Handles node and connection genes."""
import warnings
from random import random
from .attributes import FloatAttribute, BoolAttribute, StringAttribute, ListAttribute

# TODO: メタプログラミングを使ってこれらのクラスを単純化する余地は、おそらくたくさんあるだろう。
# TODO: 性能/メモリ使用量の改善のために__slot__を使用することを評価する。


class BaseGene(object):
    """
    クロスオーバーや突然変異メソッドの呼び出しなど、複数のタイプの遺伝子（ノードとコネクションの両方）で共有される機能を扱う。
    """
    def __init__(self, key: int):
        """遺伝子のキーを設定

        Args:
            key (int): 遺伝子のキー
        """
        self.key = key

    def __str__(self) -> str:
        """
        遺伝子の文字列表現を返す
        """
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other) -> bool:
        """
        遺伝子のキーを比較する
        """
        assert isinstance(self.key, type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key, other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config: dict, param_dict: dict) -> None:
        """
        コンフィグを解析して遺伝子の属性を設定する
        """
        pass

    @classmethod
    def get_config_params(cls: type) -> list:
        """
        遺伝子の属性を取得する

        Args:
            cls (type): 遺伝子のクラス

        Returns:
            list: 遺伝子の属性
        """
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__, cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('response'),
                        StringAttribute('aggregation', options='sum')]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        # d = abs(self.bias - other.bias) + abs(self.response - other.response)
        d = abs(self.response - other.response)
        # if self.activation != other.activation:
        #     d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('ws'),
                        FloatAttribute('wb'),
                        ListAttribute('ctp'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def init_attributes(self, config):
        # if hasattr(config, "ctp_num_points") and getattr(config, "ctp_num_points") > 1:
        #     for i in range(getattr(config, "ctp_num_points")): 
        #         self._gene_attributes.append(FloatAttribute('ctp' + str(i)))

        # self._gene_attributes.pop(2)

        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))
        
    def distance(self, other, config):
        # d = abs(self.weight - other.weight)
        d = abs(self.ws - other.ws) + abs(self.wb - other.wb) + sum([abs(self.ctp[i] - other.ctp[i]) for i in range(len(self.ctp))])
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
    
    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if a.name == 'ctp':
                gene1_ctp = getattr(self, 'ctp').copy()
                gene2_ctp = getattr(gene2, 'ctp').copy()
                crossover_point1 = int(random() * len(getattr(self, a.name)))
                crossover_point2 = int(random() * len(getattr(self, a.name)))
                if crossover_point1 > crossover_point2:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1

                # print(crossover_point1, crossover_point2)
                
                gene1_ctp[crossover_point1:crossover_point2] = gene2_ctp[crossover_point1:crossover_point2]
                setattr(new_gene, a.name, gene1_ctp)
            elif random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene

