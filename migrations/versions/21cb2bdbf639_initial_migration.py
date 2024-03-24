"""Initial migration

Revision ID: 21cb2bdbf639
Revises: 
Create Date: 2024-01-26 13:28:11.069867

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '21cb2bdbf639'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('hearts')
    op.drop_table('diabetics')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('diabetics',
    sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('Pregnancies', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('Glucose', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('BloodPressure', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('SkinThickness', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('Insulin', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('BMI', mysql.DECIMAL(precision=4, scale=1), nullable=False),
    sa.Column('DiabetesPedigreeFunction', mysql.DECIMAL(precision=5, scale=3), nullable=False),
    sa.Column('Age', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('Outcome', mysql.BIT(length=1), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('hearts',
    sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('age', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('sex', mysql.BIT(length=1), nullable=False),
    sa.Column('cp', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('trestbps', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('chol', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('fbs', mysql.BIT(length=1), nullable=False),
    sa.Column('restecg', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('thalach', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('exang', mysql.BIT(length=1), nullable=False),
    sa.Column('oldpeak', mysql.DECIMAL(precision=3, scale=1), nullable=False),
    sa.Column('slope', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('ca', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('thal', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('target', mysql.BIT(length=1), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    # ### end Alembic commands ###
