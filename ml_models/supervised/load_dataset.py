
import pandas as pd
from enum import Enum
from sklearn.preprocessing import LabelEncoder


class SpiType(Enum):
    DIMENSIONS = 1
    COMPONENTS = 2
    INDICATORS = 3

    
class LoadDataset:
    
    YEARS_AVAILABLE = [2014, 2018, 2021]
    
    def __init__(self):
        # TODO: Refactor to get the file from anywhere
        self.dataset = pd.read_csv("../../../data/output/hotspot_spi.csv")
    
    
    def as_daframe(self):
        """
        Get dataset as Pandas dataframe
        """
        return self.dataset
    
    
    def get_available_years(self):
        return self.YEARS_AVAILABLE
    
    
    def return_X_y_clf(self, spi_type = SpiType.INDICATORS):
        """
        Feature Selection

        X: Feature Variables (or Independent Variables)
        y: Target Variables (or dependent Variables)
        """
        X = {
            SpiType.DIMENSIONS: self.__dimensions(),
            SpiType.COMPONENTS: self.__components(),
            SpiType.INDICATORS: self.__indicators()
        }.get(spi_type)
        
        y = self.dataset["indicadoriscocat"]
        
        return (X, y)
    
    
    def return_X_y_regr(self, spi_type = SpiType.INDICATORS):
        """
        Feature Selection
        
        X: Feature Variables (or Independent Variables)
        y: Target Variables (or dependent Variables)
        """
        X = {
            SpiType.DIMENSIONS: self.__dimensions(),
            SpiType.COMPONENTS: self.__components(),
            SpiType.INDICATORS: self.__indicators()
        }.get(spi_type)

        y = LabelEncoder().fit_transform(self.dataset["indicadorisco"])

        return (X, y)
    
    
    def __dimensions(self):
        return self.dataset[["Necessidades Humanas Básicas", "Fundamentos para o Bem-Estar", "Oportunidades"]]
    
    
    def __components(self):
        return self.dataset[[
            "Nutrição e cuidados médicos básicos", "Água e saneamento","Moradia","Segurança pessoal",
            "Acesso ao conhecimento básico", "Acesso à informação e comunicação", "Saúde e bem-estar", "Qualidade do meio ambiente",
            "Direitos individuais", "Liberdade individual e de escolha", "Tolerância e inclusão", "Acesso à educação superior"
        ]]
    
    
    def __indicators(self):
        return self.dataset[[ 
            "Mortalidade infantil até 5 anos", "Mortalidade materna", "Mortalidade por desnutrição", "Mortalidade por doenças infecciosas", "Subnutrição", 
            "Abastecimento de água", "Esgotamento sanitário", "Índice de atendimento de agua",
            "Coleta de lixo", "Moradias com iluminação adequada", "Moradias com parede adequada", "Moradias com piso adequado",
            "Assassinatos de jovens", "Assassinatos de jovens Taxa", "Homicídios", "Homicídios Taxa", "Mortes por acidente no trânsito",

            "Abandono escolar ensino fundamental", "Distorção idade-série ensino fundamental", "Distorção idade-série ensino médio", "Qualidade da educação Ideb ensino fundamental", "Reprovação escolar ensino fundamental",
            "Densidade internet banda-larga", "Densidade telefonia fixa", "Densidade telefonia movel", "Densidade TV por assinatura",
            "Mortalidade por diabetes mellitus", "Mortalidade por câncer", "Mortalidade por doenças circulatórias", "Mortalidade por doenças respiratórias", "Mortalidade por suicídios",
            "Áreas Protegidas", "Desmatamento acumulado", "Desmatamento recente", "Emissões CO2", "Focos de calor por habitantes", 

            "Diversidade Partidária", "Transporte Público", 
            "Acesso à cultura, esporte e lazer", "Gravidez na infância e adolescência", "Trabalho Infantil" , "Vulnerabilidade familiar",
            "Violência contra indígenas", "Violência contra indígenas Taxa", "Violência contra mulheres", "Violência infantil", "Violência infantil Taxa",
            "Empregos ensino superior", "Mulheres com empregos ensino superior"
        ]]

    