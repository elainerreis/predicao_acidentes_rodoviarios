from pydantic import BaseModel

class AccidentInput(BaseModel):
    uf: str
    br: str
    km: float
    dia_semana: str
    fase_dia: str
    sentido_via: str
    condicao_metereologica: str
    tipo_pista: str
    tracado_via: str
    uso_solo: str
    tipo_veiculo: str
    ano: int
    frota: float
