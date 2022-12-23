from pydantic import BaseModel


class BertText(BaseModel):
    story: dict


class SummaryText(BaseModel):
    data: dict
